import os
import os.path as osp
import argparse
import time
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy
from wrapper import wrapper

from models import model_dict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train SSKD student network.')
parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t-lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[60,120,180])
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30,45])

parser.add_argument('--s-arch', type=str) # student architecture
parser.add_argument('--t-path', type=str) # teacher checkpoint path

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

t_name = osp.abspath(args.t_path).split('/')[-1]
t_arch = '_'.join(t_name.split('_')[1:-1])
exp_name = f'mpd_student_{args.arch}_time{datetime.now()}'
exp_path = './experiments/{}'.format(exp_name)
os.makedirs(exp_path, exist_ok=True)

logger = SummaryWriter(osp.join(exp_path, 'events'))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])

trainset = CIFAR100('./data', train=True, transform=transform_train)
valset = CIFAR100('./data', train=False, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

# teacher model loads checkpoint
ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
t_model = model_dict[t_arch](num_classes=100).cuda()
state_dict = torch.load(ckpt_path)['state_dict']
t_model.load_state_dict(state_dict)
t_model = wrapper(module=t_model, cfg=args).cuda()

# first train the teacher's multi-pressure tube
t_model.eval()
t_high_pressure_optimizer = optim.SGD([{'params': t_model.backbone.parameters(), 'lr': 0.0},
                                       {'params': t_model.high_pressure_encoder.parameters(), 'lr': args.t_lr},
                                       {'params': t_model.high_pressure_decoder.parameters(), 'lr': args.t_lr}
                                       ], momentum=args.momentum, weight_decay=args.weight_decay)
t_high_scheduler = MultiStepLR(t_high_pressure_optimizer, milestones=args.t_milestones, gamma=args.gamma)

t_low_pressure_optimizer = optim.SGD([{'params': t_model.backbone.parameters(), 'lr': 0.0},
                                      {'params': t_model.low_pressure_encoder.parameters(), 'lr': args.t_lr},
                                      {'params': t_model.low_pressure_decoder.parameters(), 'lr': args.t_lr}
                                      ], momentum=args.momentum, weight_decay=args.weight_decay)
t_low_scheduler = MultiStepLR(t_low_pressure_optimizer, milestones=args.t_milestones, gamma=args.gamma)


for epoch in range(args.t_epoch):
    t_model.eval()
    h_loss_record = AverageMeter()
    h_acc_record = AverageMeter()
    l_loss_record = AverageMeter()
    l_acc_record = AverageMeter()

    start = time.time()
    for img, label in train_loader:
        img = img.cuda()
        label = label.cuda()

        t_high_pressure_optimizer.zero_grad()
        t_low_pressure_optimizer.zero_grad()

        out, high_pressure_decoder_out, low_pressure_decoder_out, _ = t_model.forward(img, bb_grad=False, decoder_train=True)
        out = out.detach()

        loss_high_pressure = F.kl_div(F.log_softmax(high_pressure_decoder_out, dim=-1), F.softmax(out, dim=-1), reduction='batchmean')
        loss_high_pressure.backward()
        t_high_pressure_optimizer.step()

        loss_low_pressure = F.kl_div(F.log_softmax(low_pressure_decoder_out, dim=-1), F.softmax(out, dim=-1), reduction='batchmean')
        loss_low_pressure.backward()
        t_low_pressure_optimizer.step()

        h_acc = accuracy(high_pressure_decoder_out.data, label)[0]
        h_acc_record.update(h_acc.item(), img.size(0))
        h_loss_record.update(loss_high_pressure.item(), img.size(0))

        l_acc = accuracy(low_pressure_decoder_out.data, label)[0]
        l_acc_record.update(l_acc.item(), img.size(0))
        l_loss_record.update(loss_low_pressure.item(), img.size(0))

    logger.add_scalar('train/teacher_high_pressure_loss', h_loss_record.avg, epoch+1)
    logger.add_scalar('train/teacher_high_pressure_acc', h_acc_record.avg, epoch+1)
    logger.add_scalar('train/teacher_low_pressure_loss', l_loss_record.avg, epoch+1)
    logger.add_scalar('train/teacher_low_pressure_acc', l_acc_record.avg, epoch+1)

    run_time = time.time() - start
    msg = 'teacher train Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} hp_acc:{:.2f} lp loss:{:.3f} lp acc:{:.2f}'.format(
        epoch+1, args.t_epoch, run_time, h_loss_record.avg, h_acc_record.avg, l_loss_record.avg, l_acc_record.avg
    )
    print(msg)


    t_model.eval()
    h_loss_record = AverageMeter()
    h_acc_record = AverageMeter()
    l_loss_record = AverageMeter()
    l_acc_record = AverageMeter()

    start = time.time()
    for img, label in val_loader:
        img = img.cuda()
        label = label.cuda()

        with torch.no_grad():
            out, high_pressure_decoder_out, low_pressure_decoder_out, _ = t_model.forward(img, bb_grad=False, decoder_train=True)

        loss_high_pressure = F.kl_div(F.log_softmax(high_pressure_decoder_out, dim=-1), F.softmax(out, dim=-1), reduction='batchmean')

        loss_low_pressure = F.kl_div(F.log_softmax(low_pressure_decoder_out, dim=-1), F.softmax(out, dim=-1), reduction='batchmean')

        h_acc = accuracy(high_pressure_decoder_out.data, label)[0]
        h_acc_record.update(h_acc.item(), img.size(0))
        h_loss_record.update(loss_high_pressure.item(), img.size(0))

        l_acc = accuracy(low_pressure_decoder_out.data, label)[0]
        l_acc_record.update(l_acc.item(), img.size(0))
        l_loss_record.update(loss_low_pressure.item(), img.size(0))

    logger.add_scalar('val/teacher_high_pressure_loss', h_loss_record.avg, epoch+1)
    logger.add_scalar('val/teacher_high_pressure_acc', h_acc_record.avg, epoch+1)
    logger.add_scalar('val/teacher_low_pressure_loss', l_loss_record.avg, epoch+1)
    logger.add_scalar('val/teacher_low_pressure_acc', l_acc_record.avg, epoch+1)

    run_time = time.time() - start
    msg = 'teacher val Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} hp_acc:{:.2f} lp loss:{:.3f} lp acc:{:.2f}'.format(
        epoch+1, args.t_epoch, run_time, h_loss_record.avg, h_acc_record.avg, l_loss_record.avg, l_acc_record.avg
    )
    print(msg)

    t_high_scheduler.step()
    t_low_scheduler.step()

name = osp.join(exp_path, 'ckpt/teacher_wrapper.pth')
os.makedirs(osp.dirname(name), exist_ok=True)
torch.save(t_model.state_dict(), name)

# ----------------  start distillation ! -------------------

