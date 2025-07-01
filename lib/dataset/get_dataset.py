import os
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import ImageDraw
from dataset.randaugment import RandAugment

from dataset.handlers import (COCO2014_handler, CUB_200_2011_handler,
                              NUS_WIDE_handler, VOC2012_handler, AWA_handler)

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _safe_indexing, indexable
from sklearn.utils.validation import _num_samples
from itertools import chain

np.set_printoptions(suppress=True)

HANDLER_DICT = {
    'VOC2012': VOC2012_handler,
    'coco': COCO2014_handler,
    'nuswide': NUS_WIDE_handler,
    'cub': CUB_200_2011_handler,
    'Animals_with_Attributes2': AWA_handler

}

def multilabel_train_test_split(*arrays,
                                test_size=None,
                                train_size=None,
                                random_state=None,
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


def get_datasets(args):

    nrs = np.random.RandomState(args.seed)

    train_transform = TransformUnlabeled_WS(args)

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    source_data = load_data(args.dataset_dir)

    data_handler = HANDLER_DICT[args.dataset_name]

    train_labels = source_data['train']['labels']

    # n_train_idx = len(train_labels)
    # train_idxs_ = list(range(n_train_idx))
    # if args.dataset_name == 'coco':
    #     train_idxs, _ = multilabel_train_test_split(
    #         train_idxs_,
    #         train_size=50000,
    #         stratify=source_data['train']['labels'])
    # else:
    #     train_idxs = train_idxs_

    n_train = len(train_labels)
    # n_train = len(train_idxs)
    n_lb = int(args.lb_ratio * n_train)
    indices = nrs.permutation(n_train)
    # indices = train_idxs
    lb_idxs = indices[:n_lb]
    ub_idxs = indices[n_lb:]

    lb_train_imgs, lb_train_labels = source_data['train']['images'][
        lb_idxs], source_data['train']['labels'][lb_idxs]
    ub_train_imgs, ub_train_labels = source_data['train']['images'][
        ub_idxs], source_data['train']['labels'][ub_idxs]

    args.pos_label_freq = lb_train_labels.sum(0) / float(len(lb_train_labels))
    args.neg_label_freq = (lb_train_labels == 0).sum(0) / float(
        len(lb_train_labels))
    

    lb_train_dataset = data_handler(lb_train_imgs,
                                    lb_train_labels,
                                    args.dataset_dir,
                                    transform=train_transform)
    ub_train_dataset = data_handler(ub_train_imgs,
                                    ub_train_labels,
                                    args.dataset_dir,
                                    transform=train_transform)
    
    # n_val = len(source_data['val']['labels'])
    # val_idxs_ = list(range(n_val))
    # if args.dataset_name == 'coco':
    #     val_idxs, _ = multilabel_train_test_split(
    #         val_idxs_,
    #         train_size=10000,
    #         stratify=source_data['val']['labels'])
    # else:
    #     val_idxs = val_idxs_

    val_dataset = data_handler(source_data['val']['images'],
                               source_data['val']['labels'],
                               args.dataset_dir,
                               transform=test_transform)
        
    # val_dataset = data_handler(source_data['val']['images'][val_idxs], source_data['val']['labels'][val_idxs], args.dataset_dir, transform=test_transform)

    return lb_train_dataset, ub_train_dataset, val_dataset


def load_data(base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(
            os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['images'] = np.load(
            os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return data


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255),
                      random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class TransformUnlabeled_WS(object):
    def __init__(self, args):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor()
        ])

        strong = [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.img_size, args.img_size)),
            RandAugment(),
            transforms.ToTensor()
        ]

        if args.cutout > 0:
            strong.insert(2, CutoutPIL(cutout_factor=args.cutout))

        self.strong = transforms.Compose(strong)

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong
