import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(self,
                 gamma_neg=4,
                 gamma_pos=1,
                 clip=0.05,
                 eps=1e-8,
                 disable_torch_grad_focal_loss=True,
                 reduction='none'):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        if self.reduction == 'sum':
            loss = -loss.sum()
        elif self.reduction == 'none':
            loss = -loss

        return loss


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''
    def __init__(self,
                 gamma_neg=4,
                 gamma_pos=1,
                 clip=0.05,
                 eps=1e-8,
                 disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets *
                       torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg, self.gamma_pos * self.targets +
                self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self,
                 gamma_pos=0,
                 gamma_neg=4,
                 eps: float = 0.1,
                 reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1,
            target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes)

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


'''
loss functions
'''


def loss_an(logits, observed_labels):

    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits,
                                                     observed_labels,
                                                     reduction='none')
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(
        logits,
        torch.logical_not(observed_labels).type(logits.type()),
        reduction='none')
    return loss_matrix, corrected_loss_matrix.type(loss_matrix.type())


'''
top-level wrapper
'''


def compute_batch_loss(
        preds, label_vec,
        args):  # "preds" are actually logits (not sigmoid activated !)

    assert preds.dim() == 2

    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))

    unobserved_mask = (label_vec == 0)

    # compute loss for each image and class:
    loss_matrix, corrected_loss_matrix = loss_an(preds, label_vec.clip(0))

    correction_idx = None

    if args.clean_rate == 1:  # if epoch is 1, do not modify losses
        final_loss_matrix = loss_matrix
    else:
        if args.mod_scheme is 'LL-Cp':
            k = math.ceil(batch_size * num_classes * args.delta_rel)
        else:
            k = math.ceil(batch_size * num_classes * (1 - args.clean_rate))

        unobserved_loss = unobserved_mask.bool() * loss_matrix
        topk = torch.topk(unobserved_loss.flatten(), k)
        topk_lossvalue = topk.values[-1]
        correction_idx = torch.where(unobserved_loss > topk_lossvalue)
        if args.mod_scheme in ['LL-Ct', 'LL-Cp']:
            final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue,
                                            loss_matrix, corrected_loss_matrix)
        else:
            zero_loss_matrix = torch.zeros_like(loss_matrix)
            final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue,
                                            loss_matrix, zero_loss_matrix)

    main_loss = final_loss_matrix.sum()

    return main_loss, correction_idx


class BAMLoss(object):
    def __call__(self, outputs, targets):
        logits = torch.sigmoid(outputs)
        # L = -torch.sum(torch.log(logits) * targets + torch.log(1. - logits) *
        #                (1. - targets),
        #                dim=1)
        L = -(torch.log(logits) * targets + torch.log(1. - logits) * (1. - targets))

        return L


def applyBBAM(output, target, means, variances, mean_variances, epoch, args):
    target[target == -1] = 0
    phi = BBAM(output, target, means, variances, mean_variances, epoch, args)
    phi_theta = (phi - target * args.cos_margin) * args.cos_norm

    return phi_theta, phi


def BBAM(output, target, means, variances, mean_variances, epoch, args):
    if epoch >= args.start_bbam_epoch:
    # if epoch >= 100:
        # variances[variances < 1e-3] = 1e-3 # mean_variances
        a0 = torch.sqrt(mean_variances / variances[0])
        b0 = (1.0 - a0) * means[0]

        a1 = torch.sqrt(mean_variances / variances[1])
        b1 = (1.0 - a1) * means[1]

        theta = torch.acos(output)
        phi_theta0 = a0 * theta + b0
        phi_theta1 = a1 * theta + b1
        phi_theta = (1. - target) * phi_theta0 + target * phi_theta1
        phi = torch.cos(phi_theta)
    else:
        phi = output

    return phi


def updateBbamPara(features,
                   targets,
                   centers,
                   means,
                   variances,
                   mean_variances,
                   epoch,
                   gamma=0.1):

    centers, means, variances = calculate_mean_variance(
        features, targets, centers, means, variances, epoch, gamma)

    mean_variances = torch.mean(variances, dim=0)

    return centers, means, variances, mean_variances


def calculate_mean_variance(features,
                            targets,
                            centers,
                            means,
                            variances,
                            epoch,
                            gamma=0.1,
                            flag=0):

    targets = targets.float()
    target0 = 1. - targets
    target1 = targets
    target0[targets == -1] = 0
    target1[targets == -1] = 0

    K0 = torch.sum(target0, dim=0)
    K_var0 = K0 - 1.0
    K_var0[K_var0 <= 0.0] = 1.0

    K1 = torch.sum(target1, dim=0)
    K_var1 = K1 - 1.0
    K_var1[K_var1 <= 0.0] = 1.0

    # if flag == 0:
    #     current_centers = torch.matmul(targets.transpose(0, 1),
    #                                    features) / (K.view(-1, 1) + 1e-5)
    #     centers = (1.0 - gamma) * current_centers + gamma * centers

    centers = torch.matmul(target1.transpose(0, 1),
                           features) / (K1.view(-1, 1) + 1e-5)

    cosine_beta = torch.matmul(F.normalize(features),
                               F.normalize(centers).transpose(0, 1))
    cosine_beta = torch.clamp(cosine_beta, -1.0, 1.0)
    beta = torch.acos(cosine_beta)

    if flag == 0:
        current_means0 = torch.sum(beta * target0, dim=0) / (K0 + 1e-5)
        means[0] = (1.0 - gamma) * current_means0 + gamma * means[0]
        current_means1 = torch.sum(beta * target1, dim=0) / (K1 + 1e-5)
        means[1] = (1.0 - gamma) * current_means1 + gamma * means[1]

    current_variances0 = torch.sum(
        (beta - means[0])**2 * target0, dim=0) / K_var0
    variances[0] = (1.0 - gamma) * current_variances0 + gamma * variances[0]

    current_variances1 = torch.sum(
        (beta - means[1])**2 * target1, dim=0) / K_var1
    variances[1] = (1.0 - gamma) * current_variances1 + gamma * variances[1]

    return centers, means, variances
