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


class STLInstance(datasets.STL10):
    def __init__(self, root, split,
				 transform=None, target_transform=None,
				 download=True):
        super().__init__(root=root, split=split, download=download,
						 transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        if self.split == 'train':
            img, target = self.data[index], self.targets[index]
        elif self.split == 'test':
            img, target = self.data[index], self.targets[index]
        else: raise NotImplementedError
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_stl_dataloaders(root='data', batch_size=128, num_workers=8, is_instance=False):
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4468, 0.4399, 0.4068), (0.2419, 0.2384, 0.2541)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4468, 0.4399, 0.4068), (0.2419, 0.2384, 0.2541)),
    ])

    if is_instance:
        train_set = STLInstance(root=root,
                                download=True,
                                split='train',
                                transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.STL10(root=root,
                                      download=True,
                                      split='train',
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.STL10(root=root,
                                 download=True,
                                 split='test',
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader

#
# if __name__ == '__main__':
#     import numpy as np
#     from tqdm import tqdm
#     num = 5000
#     batch = 50
#     size = 32
#     imgs = np.zeros(shape=(num, 3, size, size))
#     train_loader, test_loader = get_stl_dataloaders('../datasets', num_workers=4, batch_size=batch)
#
#     for i, (img, _) in enumerate(tqdm(train_loader)):
#         imgs[i*batch: (i+1)*batch] = img.numpy()
#
#
#     for i in range(3):
#         c = imgs[:, i, :, :]
#         print(round(c.mean(), 4), round(c.std(), 4))