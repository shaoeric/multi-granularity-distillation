"""
get data loaders
"""
from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import torch

mean = [0.480, 0.448, 0.398]
std = [0.253, 0.245, 0.260]



class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class ImageFolderSample(datasets.ImageFolder):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.k = k
        self.is_sample = is_sample

        print('stage1 finished!')

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                path, target = self.imgs[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_test_loader(root, batch_size=128, num_workers=8):
    """get the test data loader"""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_folder = os.path.join(root, 'val')
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return test_loader


def get_dataloader_sample(root, batch_size=128, num_workers=8, is_sample=True, k=4096):
    """Data Loader for ImageNet"""


    # add data transform
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(root, 'train')
    test_folder = os.path.join(root, 'val')

    train_set = ImageFolderSample(train_folder, transform=train_transform, is_sample=is_sample, k=k)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    print('num_samples', len(train_set.samples))
    print('num_class', len(train_set.classes))

    return train_loader, test_loader, len(train_set), len(train_set.classes)




class TestImageDataset(Dataset):
    def __init__(self, root, transform, classes2label=None):
        super(TestImageDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.classes2label = classes2label
        self.image_file_list, self.label_list = self.parse_txt()

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        file = os.path.join(self.root,'images', self.image_file_list[idx])
        img = Image.open(file).convert('RGB')
        label = torch.tensor(int(self.label_list[idx])).long()

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def parse_txt(self):
        annotation_path = os.path.join(self.root, 'val_annotations.txt')
        image_file_list = []
        label_list = []

        with open(annotation_path, 'r') as f:
            contents = f.readlines()
        for content in contents:
            image_file, classes_name = content.split('\t')[:2]
            image_file_list.append(image_file)
            label = self.classes2label[classes_name]
            label_list.append(label)
        return image_file_list, label_list


def get_imagenet_dataloader(root, batch_size=128, num_workers=16, is_instance=False):
    """
    Data Loader for imagenet
    """
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(root, 'train')
    test_folder = os.path.join(root, 'val')

    if is_instance:
        train_set = ImageFolderInstance(train_folder, transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.ImageFolder(train_folder, transform=train_transform)

    test_set = TestImageDataset(root=test_folder, transform=test_transform, classes2label=train_set.class_to_idx)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader



#
# if __name__ == '__main__':
#     import numpy as np
#     from tqdm import tqdm
#     batch = 64
#     size = 32
#     train_loader, test_loader = get_imagenet_dataloader('../datasets/tiny-imagenet-200', num_workers=4,
#                                                         batch_size=batch)
#     num = len(train_loader.dataset)
#
#     imgs = np.zeros(shape=(num, 3, size, size))
#
#     for i, (img, _) in enumerate(tqdm(train_loader)):
#         imgs[i*batch: (i+1)*batch] = img.numpy()
#
#
#     for i in range(3):
#         c = imgs[:, i, :, :]
#         print(round(c.mean(), 4), round(c.std(), 4))
