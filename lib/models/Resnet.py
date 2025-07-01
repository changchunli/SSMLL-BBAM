import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def create_model(mo, n_classes):

    model = Classifier_for_Resnet(mo.lower(), n_classes)

    return model


# def resnet(mo, nc, pretrain=True):

#     if mo == 'resnet18':
#         model = models.resnet18(pretrained=pretrain)
#     elif mo == 'resnet32':
#         model = models.resnet32(pretrained=pretrain)
#     elif mo == 'resnet50':
#         model = models.resnet50(pretrained=pretrain)

#     if mo in ['resnet18','resnet32', 'resnet50']:
#         model.fc = torch.nn.Linear(model.fc.in_features, nc)

#     return model


class Classifier_for_Resnet(nn.Module):
    def __init__(self, mo, num_class=2, pretrain=True):
        super(Classifier_for_Resnet, self).__init__()
        # Load pre-trained ResNet-101 model
        if mo == 'resnet18':
            self.model = models.resnet18(pretrained=pretrain)
        elif mo == 'resnet32':
            self.model = models.resnet32(pretrained=pretrain)
        elif mo == 'resnet50':
            self.model = models.resnet50(pretrained=pretrain)

        # if mo in ['resnet18','resnet32', 'resnet50']:
        #     self.model.fc = torch.nn.Linear(self.model.fc.in_features, 256)

        # # # initialize the parameter
        # self.weight = nn.Parameter(
        #     torch.FloatTensor(num_class, 256))
        # k = math.sqrt(1.0 / self.model.fc.in_features)
        # nn.init.uniform_(self.weight, -k, k)

        self.in_features = 256 # self.model.fc.in_features
             
        if mo in ['resnet18','resnet32', 'resnet50']:
             # self.model.fc = nn.Linear(self.in_features, self.in_features)
             classifier = nn.Sequential(nn.Linear(self.model.fc.in_features, 512), nn.GELU(), nn.Linear(512, self.in_features))
             self.model.fc = classifier
 
         # # initialize the parameter
        self.weight = nn.Parameter(
             torch.FloatTensor(num_class, self.in_features))
        k = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.weight, -k, k)

    def forward(self, x, flag=0):
        feature = self.model(x)
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))

        if flag == 1:
            output = (cosine, feature)
        else:
            output = cosine
        return output
