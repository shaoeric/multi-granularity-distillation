import os
import os.path as osp
import argparse
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from data.imagenet import get_imagenet_dataloader
from data.stl import get_stl_dataloaders
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy
from models import model_dict
from wrapper import wrapper

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='train teacher network.')
parser.add_argument('--root', type=str, default='datasets')
parser.add_argument('--num_class', type=int, default=100)  # cifar pretrained num classes
parser.add_argument('--pretrained_path', type=str, default='experiments/student_best.pth')

parser.add_argument('--arch', type=str, default='resnet20')
parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])

parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[20, 25])


parser.add_argument('--kd_func', type=str, choices=['kd', 'hint', 'attention', 'similarity', 'correlation', 'vid', 'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound', 'factor', 'nst'])
parser.add_argument('--dataset', type=str, default='stl10', choices=['stl10', 'tinyimagenet'])

parser.add_argument('--print_freq', type=int, default=15)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

if args.arch in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
    args.lr = 0.01


exp_name = '{}_teacher_{}_seed{}'.format(args.kd_func, args.arch, args.seed)
exp_path = './experiments/transfer/{}/{}/{}'.format(args.dataset, exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(exp_path, exist_ok=True)
logger = SummaryWriter(osp.join(exp_path, 'events'))

print(exp_path)

if args.dataset == 'stl10':
    num_class = 10
    train_loader, val_loader = get_stl_dataloaders(root=args.root, batch_size=args.batch_size, num_workers=4)
elif args.dataset == 'tinyimagenet':
    num_class = 200
    args.batch_size = 128
    args.lr = 2 * args.lr
    train_loader, val_loader = get_imagenet_dataloader(root=args.root, batch_size=args.batch_size, num_workers=4)
else:
    raise NotImplementedError


model = model_dict[args.arch](num_classes=100)
model = wrapper(module=model, cfg=args)
model.load_state_dict(torch.load(args.pretrained_path)['state_dict'])
model = model.backbone
feat_dim = list(model.children())[-1].in_features

try:
    model.fc = nn.Linear(feat_dim, num_class)
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
except:
    try:
        model.linear = nn.Linear(feat_dim, num_class)
        optimizer = optim.SGD(model.linear.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    except:
        model.classifer = nn.Linear(feat_dim, num_class)
        optimizer = optim.SGD(model.classifer.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
model = model.cuda()

scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

best_acc = -1
for epoch in range(args.epoch):

    model.train()
    loss_record = AverageMeter()
    acc_record = AverageMeter()

    start = time.time()
    for x, target in train_loader:
        optimizer.zero_grad()
        x = x.cuda()
        target = target.cuda()

        output = model(x)
        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        batch_acc = accuracy(output, target, topk=(1,))[0]
        loss_record.update(loss.item(), x.size(0))
        acc_record.update(batch_acc.item(), x.size(0))

    logger.add_scalar('train/cls_loss', loss_record.avg, epoch + 1)
    logger.add_scalar('train/cls_acc', acc_record.avg, epoch + 1)

    run_time = time.time() - start

    info = 'train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\t'.format(
        epoch + 1, args.epoch, run_time, loss_record.avg, acc_record.avg)
    if (epoch + 1) % args.print_freq == 0:
        print(info)

    model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in val_loader:
        x = x.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(x)
            loss = F.cross_entropy(output, target)

        batch_acc = accuracy(output, target, topk=(1,))[0]
        loss_record.update(loss.item(), x.size(0))
        acc_record.update(batch_acc.item(), x.size(0))

    run_time = time.time() - start

    logger.add_scalar('val/cls_loss', loss_record.avg, epoch + 1)
    logger.add_scalar('val/cls_acc', acc_record.avg, epoch + 1)

    info = 'test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\n'.format(
        epoch + 1, args.epoch, run_time, loss_record.avg, acc_record.avg)

    if (epoch + 1) % args.print_freq == 0:
        print(info)

    scheduler.step()

    # save best
    if acc_record.avg > best_acc:
        state_dict = dict(epoch=epoch + 1, model=model.state_dict(), accuracy=acc_record.avg)
        name = osp.join(exp_path, 'ckpt/best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
        best_acc = acc_record.avg

print('student_best_acc: {:.2f}'.format(best_acc))
