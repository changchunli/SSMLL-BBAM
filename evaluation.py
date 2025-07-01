import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


def sort(x):
    temp = np.array(x)
    length = temp.shape[0]
    index = []
    sortX = []
    for i in range(length):
        Min = float("inf")
        Min_j = i
        for j in range(length):
            if temp[j] < Min:
                Min = temp[j]
                Min_j = j
        sortX.append(Min)
        index.append(Min_j)
        temp[Min_j] = float("inf")
    return sortX, index


def avgprec(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)
            labels_index.append(index1)
            not_labels_index.append(index2)

    aveprec = 0
    for i in range(instance_num):
        tempvalue, index = sort(temp_outputs[i])
        indicator = np.zeros((class_num, ))
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            indicator[loc] = 1
        summary = 0
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            # print(loc)
            summary = summary + sum(
                indicator[loc:class_num]) * 1.0 / (class_num - loc)
        aveprec = aveprec + summary * 1.0 / labels_size[i]
    return aveprec * 1.0 / test_data_num


def ranking_metrics_ml(test_target, predict_score):
    (test_ins_num, label_num) = np.shape(test_target)

    coverage = (metrics.coverage_error(test_target, predict_score) -
                1) / label_num
    rloss = metrics.label_ranking_loss(test_target, predict_score)
    ravgprec = metrics.label_ranking_average_precision_score(
        test_target, predict_score)
    mAP = function_mAP(test_target, predict_score)
    # pr_auc = metrics.average_precision_score(test_target, predict_score)
    pr_auc = MacroAP(predict_score, test_target)
    try:
        roc_auc = metrics.roc_auc_score(test_target, predict_score)
    except ValueError:
        roc_auc = 0

    return coverage, rloss, ravgprec, mAP, pr_auc, roc_auc


def function_mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    average_precision_list = []

    for j in range(preds.shape[1]):
        average_precision_list.append(compute_avg_precision(targs[:, j], preds[:, j]))

    return 100.0 * float(np.mean(average_precision_list))


def compute_avg_precision(targs, preds):

    '''
    Compute average precision.

    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    '''

    check_inputs(targs, preds)

    if np.all(targs == 0):
        # If a class has zero true positives, we define average precision to be zero.
        metric_value = 0.0
    else:
        metric_value = metrics.average_precision_score(targs, preds)

    return metric_value


def check_inputs(targs, preds):
    assert (np.shape(preds) == np.shape(targs))
    assert type(preds) is np.ndarray
    assert type(targs) is np.ndarray
    assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
    assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
    assert (len(np.unique(targs)) <= 2)


def binary_prediction_metrics_ml(test_target, predict_label):
    macro_f1 = metrics.f1_score(test_target, predict_label, average='macro')
    micro_f1 = metrics.f1_score(test_target, predict_label, average='micro')
    hloss = metrics.hamming_loss(test_target, predict_label)
    subset_accuracy = metrics.accuracy_score(test_target, predict_label)

    return macro_f1, micro_f1, hloss


def MacroAUC(outputs, test_target):
    # auc = metrics.roc_auc_score(test_target, outputs, average='macro')
    label_num = outputs.shape[1]
    auc = 0
    count = 0
    for i in range(label_num):
        if sum(test_target[:, i]) != 0:
            auc += metrics.roc_auc_score(test_target[:, i], outputs[:, i])
            count += 1
    auc = auc / count
    return auc


def MicroAUC(outputs, test_target):
    auc = metrics.roc_auc_score(test_target, outputs, average='micro')
    return auc


def MacroAP(outputs, test_target):
    label_num = outputs.shape[1]
    auc = 0
    count = 0
    for i in range(label_num):
        if sum(test_target[:, i]) != 0:
            auc += metrics.average_precision_score(test_target[:, i],
                                                   outputs[:, i])
            count += 1
        else:
            count += 1
    auc = auc / count
    return auc


def one_error(test_target, Outputs):
    (test_ins_num, label_num) = np.shape(test_target)
    max_score = np.max(Outputs, axis=1)
    flag_false = [
        0 if
        (test_target[i][np.where(Outputs[i] == max_score[i])[0]]).any() else 1
        for i in range(test_ins_num)
    ]
    oneerr = np.sum(flag_false) / test_ins_num

    top_label = np.argmax(Outputs, axis=1)
    num_label = [np.sum(top_label == i) for i in range(label_num)]
    top_label[top_label == 0] = -2
    top_false_label = np.array(flag_false) * top_label
    top_false_label[top_false_label == 0] = -1
    top_false_label[top_false_label == -2] = 0
    num_false_label = [np.sum(top_false_label == i) for i in range(label_num)]

    return oneerr


def classification_metrics_ml(Outputs, test_target, avg_num_rel=-1):

    (test_ins_num, label_num) = np.shape(test_target)

    predict_label_idxes = []
    if avg_num_rel != -1 and avg_num_rel != 0:
        predict_label_idxes = np.argsort(-1.0 * Outputs)[:, :avg_num_rel]
    elif avg_num_rel == 0:
        label_idxes = np.arange(label_num)
        predict_label_idxes = [
            label_idxes[Outputs[i] > 0.5] for i in range(test_ins_num)
        ]
    elif avg_num_rel == -1:
        predict_label_idxes = [
            np.argsort(-1.0 * Outputs[i])[:int(np.sum(test_target[i]))]
            for i in range(test_ins_num)
        ]

    predict_label = np.zeros(np.shape(test_target))
    for i in range(test_ins_num):
        predict_label[i][predict_label_idxes[i]] = 1

    coverage, rloss, ravgprec, mAP, pr_auc, roc_auc = ranking_metrics_ml(
        test_target, Outputs)

    macro_f1, micro_f1, hloss = binary_prediction_metrics_ml(
        test_target, predict_label)
    try:
        macro_auc = MacroAUC(predict_label, test_target)
    except ValueError:
        macro_auc = 0
    micro_auc = MicroAUC(predict_label, test_target)
    one_loss = one_error(test_target, Outputs)

    return (coverage, rloss, ravgprec, pr_auc, roc_auc, macro_f1, micro_f1,
            hloss, macro_auc, micro_auc, one_loss, mAP)


def classification_metrics(labels_truth, labels_predicted):
    macro_f1 = f1_score(labels_truth, labels_predicted, average='macro')
    micro_f1 = f1_score(labels_truth, labels_predicted, average='micro')
    (precision, recall, fbeta_score,
     support) = precision_recall_fscore_support(labels_truth, labels_predicted)

    print("precision:\n{}\nrecall:\n{}\nfbeta_score:\n{}\nsupport:\n{}".format(
        precision, recall, fbeta_score, support))
    print("macro f1: {}\tmicro f1: {}".format(macro_f1, micro_f1))

    return macro_f1, micro_f1
