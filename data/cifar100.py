from __future__ import print_function

import os
import socket
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __init__(self, root, train=True,
				 transform=None, target_transform=None,
				 download=True):
        super().__init__(root=root, train=train, download=download,
						 transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(root='data', batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=root,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR100(root=root,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=root,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class CIFAR100IdxSample(datasets.CIFAR100):
	def __init__(self, root, train=True, is_sample=True,
				 transform=None, target_transform=None,
				 download=False, n=4096, mode='exact', percent=1.0):
		super().__init__(root=root, train=train, download=download,
						 transform=transform, target_transform=target_transform)
		self.n = n
		self.mode = mode
		self.sample = is_sample

		num_classes = 100
		num_samples = len(self.data)
		labels = self.targets

		self.cls_positive = [[] for _ in range(num_classes)]
		for i in range(num_samples):
			self.cls_positive[labels[i]].append(i)

		self.cls_negative = [[] for _ in range(num_classes)]
		for i in range(num_classes):
			for j in range(num_classes):
				if j == i:
					continue
				self.cls_negative[i].extend(self.cls_positive[j])

		self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
		self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

		if 0 < percent < 1:
			num = int(len(self.cls_negative[0]) * percent)
			self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:num]
								 for i in range(num_classes)]

		self.cls_positive = np.asarray(self.cls_positive)
		self.cls_negative = np.asarray(self.cls_negative)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		if not self.sample:
			return img, target, index

		if self.mode == 'exact':
			pos_idx = index
		elif self.mode == 'relax':
			pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
		else:
			raise NotImplementedError(self.mode)
		replace = True if self.n > len(self.cls_negative[target]) else False
		neg_idx = np.random.choice(self.cls_negative[target], self.n, replace=replace)
		sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

		return img, target, index, sample_idx


def get_cifar100_dataloaders_sample(root, batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cifar 100
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100IdxSample(root=root,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       n=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=root,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
