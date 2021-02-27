from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from models import model_dict
from utils import AverageMeter
import torch
import torch.nn.functional as F
from wrapper import wrapper

parser = argparse.ArgumentParser(description='KL and CE analysis on cifar100')
parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])
parser.add_argument('--t_arch', type=str, required=True)
parser.add_argument('--s_arch', type=str, required=True)
parser.add_argument('--t_weight', type=str, required=True)
parser.add_argument('--s_weight', type=str, required=True)
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_set = datasets.CIFAR100(root=args.root, download=True, train=False, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

# teacher and student model
teacher = model_dict[args.t_arch](num_classes=100).to(args.device)
teacher.load_state_dict(torch.load(args.t_weight))
student = model_dict[args.s_arch](num_classes=100)
student = wrapper(student, args).to(args.device)
student.load_state_dict(torch.load(args.s_weight))