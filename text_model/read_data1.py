import pickle
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _safe_indexing, indexable
from sklearn.utils.validation import _num_samples
from torch.utils.data import Dataset

from pytorch_transformers import *


def multilabel_train_test_split(*arrays,
                                test_size=None,
                                train_size=None,
                                random_state=123,
                                shuffle=True,
                                stratify=None):
    """
    Train test split for multilabel classification. Uses the algorithm from:
    'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.
    """
    if stratify is None:
        return train_test_split(*arrays,
                                test_size=test_size,
                                train_size=train_size,
                                random_state=random_state,
                                stratify=None,
                                shuffle=shuffle)

    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"

    n_arrays = len(arrays)
    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(n_samples,
                                              test_size,
                                              train_size,
                                              default_test_size=0.25)
    cv = MultilabelStratifiedShuffleSplit(test_size=n_test,
                                          train_size=n_train,
                                          random_state=123)
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(
        chain.from_iterable((_safe_indexing(a, train), _safe_indexing(a, test))
                            for a in arrays))


class Translator:
    """Backtranslation. Here to save time, we pre-processing and save all the translated data into pickle files.
    """
    def __init__(self, path, transform_type='BackTranslation'):
        # Pre-processed German data
        with open(path + 'de_uda.pkl', 'rb') as f:
            self.de = pickle.load(f)
        # # Pre-processed Russian data
        # with open(path + 'ru_1.pkl', 'rb') as f:
        #     self.ru = pickle.load(f)

    def __call__(self, ori, idx):
        out1 = self.de[idx]
        # out2 = self.ru[idx]
        # return out1, out2, ori
        return out1, ori


def read_mltc(data_path):
    with open(data_path, 'r', encoding='utf8') as data_file:
        data_lines = data_file.readlines()

    labels, texts = [], []
    for line in data_lines:
        line = line.strip('\n').split('\t')
        labels.append([int(i) for i in list(line[0])])
        texts.append(line[1])

    return texts, labels


def get_data_mltc(args,
                  max_seq_len=256,
                  model='bert-base-uncased',
                  train_aug=False):
    """Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
        rate_labeled {float} -- rate of labeled data

    Keyword Arguments:
        rate_unlabeled {float} -- rate of unlabeled data (default: {0.5})
        max_seq_len {int} -- Maximum sequence length (default: {256})
        model {str} -- Model name (default: {'bert-base-uncased'})
        train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

    """

    data_path = args.dataset_dir
    # num_labeled, num_unlabeled = args.num_labeled, args.num_unlabeled
    model = args.net

    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(model)

    train_texts, train_labels = read_mltc(data_path + '/train.tsv')
    val_texts, val_labels = read_mltc(data_path + '/dev.tsv')
    test_texts, test_labels = read_mltc(data_path + '/test.tsv')

    train_texts, train_labels = np.asarray(train_texts, dtype=object), np.array(train_labels)
    val_texts, val_labels = np.asarray(val_texts, dtype=object), np.array(val_labels)
    test_texts, test_labels = np.asarray(test_texts, dtype=object), np.array(test_labels)

    n_labels = train_labels.shape[1]

    num_labeled = int(args.lb_ratio * train_labels.shape[0])
    num_unlabeled = train_labels.shape[0] - num_labeled - 1
    # num_labeled = 2000
    # num_unlabeled = 40000

    train_idxs = list(range(len(train_labels)))
    train_labeled_idxs, train_unlabeled_idxs_temp = multilabel_train_test_split(
        train_idxs,
        train_size=num_labeled,
        stratify=None)#train_labels

    train_unlabeled_idxs, _ = multilabel_train_test_split(
        train_unlabeled_idxs_temp,
        train_size=num_unlabeled,
        stratify=None)#train_labels[train_unlabeled_idxs_temp]

    # train_labeled_idxs = train_idxs
    # train_unlabeled_idxs = train_idxs

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(train_texts[train_labeled_idxs],
                                           train_labels[train_labeled_idxs],
                                           tokenizer, max_seq_len, train_aug)

    # train_unlabeled_dataset = loader_unlabeled(
    #     train_texts[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer,
    #     max_seq_len)
    train_unlabeled_dataset = loader_labeled(
        train_texts[train_unlabeled_idxs], train_labels[train_unlabeled_idxs],
        tokenizer, max_seq_len, train_aug)

    val_dataset = loader_labeled(val_texts, val_labels, tokenizer, max_seq_len)
    test_dataset = loader_labeled(test_texts, test_labels, tokenizer,
                                  max_seq_len)

    train_labeled_labels = train_labeled_dataset.get_labels()
    args.pos_label_freq = train_labeled_labels.sum(0) / float(
        len(train_labeled_labels))
    args.neg_label_freq = (1. - train_labeled_labels).sum(0) / float(
        len(train_labeled_labels))

    # print(args.pos_label_freq)

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(
        len(train_labeled_idxs), len(train_unlabeled_idxs), len(val_labels),
        len(test_labels)))

    return (train_labeled_dataset, train_unlabeled_dataset, val_dataset,
            test_dataset, n_labels)


def get_data(args,
             max_seq_len=256,
             model='bert-base-uncased',
             train_aug=False):
    """Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
        rate_labeled {float} -- rate of labeled data

    Keyword Arguments:
        rate_unlabeled {float} -- rate of unlabeled data (default: {0.5})
        max_seq_len {int} -- Maximum sequence length (default: {256})
        model {str} -- Model name (default: {'bert-base-uncased'})
        train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

    """

    data_path = args.data_path
    num_labeled, num_unlabeled = args.num_labeled, args.num_unlabeled
    model = args.model

    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(model)

    labeled_df = pd.read_csv(data_path + 'train_labeled.csv', header=None)
    unlabeled_df = pd.read_csv(data_path + 'train_unlabeled.csv', header=None)
    val_df = pd.read_csv(data_path + 'train_val.csv', header=None)

    test_df = pd.read_csv(data_path + 'test.csv', header=None)

    # Here we only use the bodies and removed titles to do the classifications
    labeled_labels = np.array([v - 1 for v in labeled_df[0]])
    labeled_text = np.array([v for v in labeled_df[2]])
    train_labeled_idxs, _ = train_test_split(list(range(len(labeled_labels))),
                                             train_size=num_labeled,
                                             stratify=labeled_labels)

    n_labels = max(labeled_labels) + 1

    unlabeled_labels = np.array([v - 1 for v in unlabeled_df[0]])
    unlabeled_text = np.array([v for v in unlabeled_df[2]])
    unlabeled_idxs = list(range(len(unlabeled_labels)))
    train_unlabeled_idxs, _ = train_test_split(unlabeled_idxs,
                                               train_size=num_unlabeled,
                                               stratify=unlabeled_labels)
    # idxs_dict = {}
    # for v in range(len(unlabeled_idxs)):
    #     idxs_dict[unlabeled_idxs[v]] = v
    # train_unlabeled_idxs1 = [idxs_dict[v] for v in train_unlabeled_idxs]

    val_labels_full = np.array([v - 1 for v in val_df[0]])
    val_text_full = np.array([v for v in val_df[2]])
    val_idxs_full = list(range(len(val_labels_full)))
    val_idxs, _ = train_test_split(val_idxs_full,
                                   train_size=n_labels * 1000,
                                   stratify=val_labels_full)

    test_labels = np.array([v - 1 for v in test_df[0]])
    test_text = np.array([v for v in test_df[2]])

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(labeled_text[train_labeled_idxs],
                                           labeled_labels[train_labeled_idxs],
                                           tokenizer, max_seq_len, train_aug)

    # train_unlabeled_dataset = loader_unlabeled(
    #     unlabeled_text[train_unlabeled_idxs1], train_unlabeled_idxs, tokenizer,
    #     max_seq_len, Translator(data_path))

    # train_unlabeled_dataset = loader_unlabeled(
    #     unlabeled_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer,
    #     max_seq_len)
    train_unlabeled_dataset = loader_labeled(
        unlabeled_text[train_unlabeled_idxs], unlabeled_labels[train_unlabeled_idxs],
        tokenizer, max_seq_len, train_aug)


    val_dataset = loader_labeled(val_text_full[val_idxs],
                                 val_labels_full[val_idxs], tokenizer,
                                 max_seq_len)
    test_dataset = loader_labeled(test_text, test_labels, tokenizer,
                                  max_seq_len)

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(
        len(train_labeled_idxs), len(train_unlabeled_idxs), len(val_idxs),
        len(test_labels)))

    return (train_labeled_dataset, train_unlabeled_dataset, val_dataset,
            test_dataset, n_labels)


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self,
                 dataset_text,
                 dataset_label,
                 tokenizer,
                 max_seq_len,
                 aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load('pytorch/fairseq',
                                        'transformer.wmt19.en-de.single_model',
                                        tokenizer='moses',
                                        bpe='fastbpe')
            self.de2en = torch.hub.load('pytorch/fairseq',
                                        'transformer.wmt19.de-en.single_model',
                                        tokenizer='moses',
                                        bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text, sampling=True, temperature=0.9),
                                                         sampling=True,
                                                         temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)),
                    (self.labels[idx], self.labels[idx]), (text_length,
                                                           text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_unlabeled(Dataset):
    # Data loader for unlabeled data
    def __init__(self,
                 dataset_text,
                 unlabeled_idxs,
                 tokenizer,
                 max_seq_len,
                 aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    # def __getitem__(self, idx):
    #     if self.aug is not None:
    #         u, v, ori = self.aug(self.text[idx], self.ids[idx])
    #         encode_result_u, length_u = self.get_tokenized(u)
    #         encode_result_v, length_v = self.get_tokenized(v)
    #         encode_result_ori, length_ori = self.get_tokenized(ori)
    #         return ((torch.tensor(encode_result_u),
    #                  torch.tensor(encode_result_v),
    #                  torch.tensor(encode_result_ori)), (length_u, length_v,
    #                                                     length_ori))
    #     else:
    #         text = self.text[idx]
    #         encode_result, length = self.get_tokenized(text)
    #         return (torch.tensor(encode_result), length)

    def __getitem__(self, idx):
        if self.aug is not None:
            u, ori = self.aug(self.text[idx], self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u),
                     torch.tensor(encode_result_ori)), (length_u, length_ori))
        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)


class loader_labeled_mask(Dataset):
    # Data loader for labeled data
    def __init__(self,
                 dataset_text,
                 dataset_label,
                 dataset_mask,
                 tokenizer,
                 max_seq_len,
                 aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.mask = dataset_mask
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load('pytorch/fairseq',
                                        'transformer.wmt19.en-de.single_model',
                                        tokenizer='moses',
                                        bpe='fastbpe')
            self.de2en = torch.hub.load('pytorch/fairseq',
                                        'transformer.wmt19.de-en.single_model',
                                        tokenizer='moses',
                                        bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text, sampling=True, temperature=0.9),
                                                         sampling=True,
                                                         temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)),
                    (self.labels[idx], self.labels[idx]),
                    (text_length, text_length2), self.mask[idx])
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length,
                    self.mask[idx])
        
class loader_labeled_mm(Dataset):
    # Data loader for labeled data
    def __init__(self,
                 dataset_text,
                 dataset_label,
                 dataset_mask,
                 dataset_mask_ns,
                 tokenizer,
                 max_seq_len,
                 aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.mask = dataset_mask
        self.mask_ns = dataset_mask_ns
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load('pytorch/fairseq',
                                        'transformer.wmt19.en-de.single_model',
                                        tokenizer='moses',
                                        bpe='fastbpe')
            self.de2en = torch.hub.load('pytorch/fairseq',
                                        'transformer.wmt19.de-en.single_model',
                                        tokenizer='moses',
                                        bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text, sampling=True, temperature=0.9),
                                                         sampling=True,
                                                         temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)),
                    (self.labels[idx], self.labels[idx]),
                    (text_length, text_length2), self.mask[idx])
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length,
                    self.mask[idx], self.mask_ns[idx])
