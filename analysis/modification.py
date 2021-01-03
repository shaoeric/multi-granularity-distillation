#### 探究迁移知识的多少是否也对学生有影响

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from models import model_dict
from utils import AverageMeter
import torch.nn.functional as F

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--t_arch', type=str, required=True)
parser.add_argument('--s_arch', type=str, required=True)
parser.add_argument('--t_weight', type=str, required=True)
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()


teacher = model_dict[args.t_arch](num_classes=100).to(args.device)
student = model_dict[args.s_arch](num_classes=100).to(args.device)


if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
elif args.dataset == 'cifar100':
    mean = (0.5071, 0.4866, 0.4409)
    std = (0.2675, 0.2565, 0.2761)
else:
    raise NotImplementedError

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])

trainset = datasets.CIFAR100('./data', train=True, transform=transform_train)
valset = datasets.CIFAR100('./data', train=False, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)


