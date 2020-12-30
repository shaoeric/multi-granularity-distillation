import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from models import model_dict
from utils import AverageMeter
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--t_arch', type=str, required=True)
parser.add_argument('--s_arch', type=str, required=True)
# parser.add_argument('--t_weight', type=str, required=True)
parser.add_argument('--root', type=str, default='../data')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

torch.manual_seed(0)

def compute_noise(out):
    out = torch.softmax(out, dim=1)
    max_value, index = out.max(dim=1, keepdim=True)
    reserve_index = out.eq(max_value)
    mask = reserve_index.bitwise_not()
    noise_compo = out.masked_select(mask)  # 输出的信息向量中 除最大值以外的输出值

    base = (1.0 - max_value) / (out.size(1)-1)  # 每个样本的噪声基础分布
    base = base.repeat(1, 3).view(out.size(0)*3, )
    return torch.sum(torch.abs(noise_compo - base)) / out.size(0)



teacher = model_dict[args.t_arch](num_classes=100).to(args.device)
# student = model_dict[args.s_arch](num_classes=100).to(args.device)

if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
elif args.dataset == 'cifar100':
    mean = (0.5071, 0.4866, 0.4409)
    std = (0.2675, 0.2565, 0.2761)
else:
    raise NotImplementedError


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])

valset = datasets.CIFAR100('../data', train=False, transform=transform_test)
val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)

noise_recorder = AverageMeter()
with torch.no_grad():
    for img, _ in val_loader:
        img = img.to(args.device)
        out = teacher.forward(img)
        noise = compute_noise(out)
        noise_recorder.update(noise, img.size(0))

print(f"model noise:{noise_recorder.avg}")