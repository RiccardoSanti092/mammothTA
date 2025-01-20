

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR100


from datasets.seq_cifar100 import TCIFAR100, MyCIFAR100
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args
from utils.prompt_templates import templates


class SequentialCIFAR100224(ContinualDataset):
    """
    The Sequential CIFAR100 dataset with 224x224 resolution with ViT-B/16.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data.
        TEST_TRANSFORM (torchvision.transforms): transformation to apply to the test data.
    """

    NAME = 'seq-cifar100-224'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = 100
    SIZE = (224, 224)
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def __init__(self, args, transform_type: str = 'weak'):
        super().__init__(args)

        assert transform_type in ['weak', 'strong'], "Transform type must be either 'weak' or 'strong'."

        if transform_type == 'strong':
            logging.info("Using strong augmentation for CIFAR100-224")
            self.TRANSFORM = transforms.Compose(
                [transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                 transforms.RandomRotation(15),
                 transforms.ToTensor(),
                 transforms.Normalize(SequentialCIFAR100224.MEAN, SequentialCIFAR100224.STD)]
            )

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM

        test_transform = self.TEST_TRANSFORM

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                   download=True, transform=transform)
        test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False,
                                 download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100224.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR100224.MEAN, SequentialCIFAR100224.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR100224.MEAN, SequentialCIFAR100224.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR100(base_path() + 'CIFAR100', train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']


class SequentialCIFAR100224_5(SequentialCIFAR100224):
    """
    Subclass of SequentialCIFAR100224 with updated settings:
        - NAME: 'seq-cifar100-224-5'
        - N_CLASSES_PER_TASK: 20
        - N_TASKS: 5
    """

    NAME = 'seq-cifar100-224-5'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5


    def __init__(self, args, transform_type: str = 'weak'):
        super().__init__(args, transform_type)

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"


class SequentialCIFAR100224_5_permutato(SequentialCIFAR100224):
    """
    Subclass of SequentialCIFAR100224 with updated settings:
        - NAME: 'seq-cifar100-224-5-permutato'
        - N_CLASSES_PER_TASK: 20
        - N_TASKS: 5
    """

    NAME = 'seq-cifar100-224-5-permutato'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5
    targets = np.array([70, 89, 11, 13, 63, 53, 86, 57, 41, 43, 14, 98, 52, 73, 95, 96, 33, 16, 39, 74, 25, 88,
                           35, 28, 79, 82, 72, 4, 30, 17, 59, 97, 36, 38, 29, 55, 83, 7, 22, 48, 19, 47, 2, 44, 67,
                           71, 34, 84, 6, 46, 61, 8, 80, 10, 49, 15, 68, 9, 99, 40, 27, 45, 51, 37, 21, 64, 92, 24,
                           60, 31, 5, 91, 93, 90, 65, 66, 77, 20, 58, 62, 23, 76, 75, 42, 0, 26, 87, 50, 3, 56, 81,
                           1, 94, 69, 18, 78, 54, 12, 85, 32])
    targets1 = np.array([26, 46, 48, 95, 53, 77, 36, 96, 1, 5, 99, 86, 30, 50, 78, 15, 54, 49, 18, 62, 45, 76, 65, 14,
                         57, 37, 16, 31, 3, 32, 22, 43, 28, 92, 27, 88, 11, 71, 69, 91, 6, 94, 29, 40, 25, 23, 35, 97,
                         12, 87, 33, 83, 68, 47, 13, 58, 2, 85, 4, 51, 67, 39, 79, 72, 10, 7, 42, 38, 21, 60, 80, 41,
                         44, 90, 17, 20, 19, 89, 55, 59, 81, 56, 61, 8, 66, 0, 75, 24, 70, 64, 74, 73, 63, 93, 9,
                         82, 34, 84, 52, 98])

    def __init__(self, args, transform_type: str = 'weak'):
        args.class_order = self.targets1
        super().__init__(args, transform_type)


    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"
