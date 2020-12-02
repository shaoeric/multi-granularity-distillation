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

# python student.py --s-arch resnet20 --t-path ./experiments/teacher_resnet56_seed0/

parser = argparse.ArgumentParser(description='train SSKD student network.')
parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])
parser.add_argument('--alpha', type=float, default=0.2)

parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t-lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[60,120,180])
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30,45])

parser.add_argument('--s-arch', type=str) # student architecture
parser.add_argument('--t-arch', type=str) # teacher architecture
parser.add_argument('--t-path', type=str) # teacher checkpoint path
parser.add_argument('--T', type=float, default=2.0) # temperature

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=40)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


t_arch = args.t_arch
exp_name = f'mpd_T_{t_arch}_S_{args.s_arch}'
exp_path = './experiments/{}/{}'.format(exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(exp_path, exist_ok=True)

logger = SummaryWriter(osp.join(exp_path, 'events'))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
state_dict = torch.load(ckpt_path)['model']
t_model.load_state_dict(state_dict)
t_model = wrapper(module=t_model, cfg=args).cuda()

# first train the teacher's multi-pressure tube
t_model.eval()
t_high_pressure_optimizer = optim.SGD([{'params': t_model.high_pressure_encoder.parameters(), 'lr': args.t_lr},
                                       {'params': t_model.high_pressure_decoder.parameters(), 'lr': args.t_lr}
                                       ], momentum=args.momentum, weight_decay=args.weight_decay)
t_high_scheduler = MultiStepLR(t_high_pressure_optimizer, milestones=args.t_milestones, gamma=args.gamma)


t_low_pressure_optimizer = optim.SGD([{'params': t_model.low_pressure_encoder.parameters(), 'lr': args.t_lr},
                                      {'params': t_model.low_pressure_decoder.parameters(), 'lr': args.t_lr}
                                      ], momentum=args.momentum, weight_decay=args.weight_decay)
t_low_scheduler = MultiStepLR(t_low_pressure_optimizer, milestones=args.t_milestones, gamma=args.gamma)


high_pressure_state_dict = osp.join(exp_path, 'ckpt/teacher_high_encoder_best.pth')
low_pressure_state_dict = osp.join(exp_path, 'ckpt/teacher_low_encoder_best.pth')


def teacher_eval(t_model, val_loader):
    t_model.eval()
    h_loss_record = AverageMeter()
    h_acc_record = AverageMeter()
    l_loss_record = AverageMeter()
    l_acc_record = AverageMeter()

    for img, label in val_loader:
        img = img.cuda()
        label = label.cuda()

        with torch.no_grad():
            out, high_pressure_decoder_out, low_pressure_decoder_out, _ = t_model.forward(img,
                                                                                          bb_grad=False,
                                                                                          decoder_train=True)

        loss_high_pressure = F.kl_div(F.log_softmax(high_pressure_decoder_out / 2.0, dim=-1),
                                      F.softmax(out / 2.0, dim=-1),
                                      reduction='batchmean') * 2.0 * 2.0 + F.cross_entropy(
            high_pressure_decoder_out, label)
        loss_low_pressure = F.kl_div(F.log_softmax(low_pressure_decoder_out / 8.0, dim=-1),
                                     F.softmax(out / 8.0, dim=-1),
                                     reduction='batchmean') * 8.0 * 8.0 + F.cross_entropy(
            low_pressure_decoder_out, label)

        h_acc = accuracy(high_pressure_decoder_out.data, label)[0]
        h_acc_record.update(h_acc.item(), img.size(0))
        h_loss_record.update(loss_high_pressure.item(), img.size(0))

        l_acc = accuracy(low_pressure_decoder_out.data, label)[0]
        l_acc_record.update(l_acc.item(), img.size(0))
        l_loss_record.update(loss_low_pressure.item(), img.size(0))
    return h_loss_record, h_acc_record, l_loss_record, l_acc_record


def student_eval(t_model, s_model, val_loader):
    s_model.eval()
    s_high_pressure_loss_record = AverageMeter()
    s_low__pressure_loss_record = AverageMeter()
    s_logits_loss_record = AverageMeter()
    s_acc_record = AverageMeter()


    for img, target in val_loader:
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            t_out, t_high_pressure_encoder_out, t_low_pressure_encoder_out, _ = t_model.forward(img, bb_grad=False, decoder_train=False)
            s_out, s_high_pressure_encoder_out, s_low_pressure_encoder_out, _ = s_model.forward(img, bb_grad=False, decoder_train=False)

        logits_loss = F.kl_div(
            F.log_softmax(s_out / 4.0, dim=1),
            F.softmax(t_out / 4.0, dim=1),
            reduction='batchmean'
        ) * 4.0 * 4.0 * (1 - args.alpha)+ F.cross_entropy(s_out, target) * args.alpha

        high_loss = F.kl_div(
            F.log_softmax(s_high_pressure_encoder_out / 2.0, dim=1),
            F.softmax(t_high_pressure_encoder_out / 2.0, dim=1),
            reduction='batchmean'
        ) * 2.0 * 2.0

        low_loss = F.kl_div(
            F.log_softmax(s_low_pressure_encoder_out / 8.0, dim=1),
            F.softmax(t_low_pressure_encoder_out / 8.0, dim=1),
            reduction='batchmean'
        ) * 8.0 * 8.0

        s_high_pressure_loss_record.update(high_loss.item(), img.size(0))
        s_low__pressure_loss_record.update(low_loss.item(), img.size(0))
        s_logits_loss_record.update(logits_loss.item(), img.size(0))
        acc = accuracy(s_out.data, target)[0]
        s_acc_record.update(acc.item(), img.size(0))
    return s_high_pressure_loss_record, s_logits_loss_record, s_low__pressure_loss_record, s_acc_record

best_low_acc = 0
best_high_acc = 0
for epoch in range(args.t_epoch):
    t_model.eval()
    h_loss_record = AverageMeter()
    h_acc_record = AverageMeter()
    l_loss_record = AverageMeter()
    l_acc_record = AverageMeter()

    start = time.time()
    # train high pressure
    for img, label in train_loader:
        img = img.cuda()
        label = label.cuda()

        t_high_pressure_optimizer.zero_grad()

        out, high_pressure_decoder_out, _, _ = t_model.forward(img, bb_grad=False, decoder_train=True)
        out = out.detach()

        loss_high_pressure = F.kl_div(F.log_softmax(high_pressure_decoder_out / 2.0, dim=-1), F.softmax(out / 2.0, dim=-1), reduction='batchmean') * 2.0 * 2.0 + F.cross_entropy(high_pressure_decoder_out, label)
        loss_high_pressure.backward()
        t_high_pressure_optimizer.step()

        h_acc = accuracy(high_pressure_decoder_out.data, label)[0]
        h_acc_record.update(h_acc.item(), img.size(0))
        h_loss_record.update(loss_high_pressure.item(), img.size(0))

    # train low pressure
    for img, label in train_loader:
        img = img.cuda()
        label = label.cuda()

        t_low_pressure_optimizer.zero_grad()

        out, _, low_pressure_decoder_out, _ = t_model.forward(img, bb_grad=False, decoder_train=True)
        out = out.detach()

        loss_low_pressure = F.kl_div(F.log_softmax(low_pressure_decoder_out / 8.0, dim=-1), F.softmax(out / 8.0, dim=-1), reduction='batchmean') * 8.0 * 8.0 + F.cross_entropy(low_pressure_decoder_out, label)
        loss_low_pressure.backward()
        t_low_pressure_optimizer.step()

        l_acc = accuracy(low_pressure_decoder_out.data, label)[0]
        l_acc_record.update(l_acc.item(), img.size(0))
        l_loss_record.update(loss_low_pressure.item(), img.size(0))

    logger.add_scalar('t_train/teacher_high_pressure_loss', h_loss_record.avg, epoch+1)
    logger.add_scalar('t_train/teacher_high_pressure_acc', h_acc_record.avg, epoch+1)
    logger.add_scalar('t_train/teacher_low_pressure_loss', l_loss_record.avg, epoch+1)
    logger.add_scalar('t_train/teacher_low_pressure_acc', l_acc_record.avg, epoch+1)

    run_time = time.time() - start
    msg = 'teacher train Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} hp_acc:{:.2f} lp loss:{:.3f} lp_acc:{:.2f}'.format(
        epoch+1, args.t_epoch, run_time, h_loss_record.avg, h_acc_record.avg, l_loss_record.avg, l_acc_record.avg
    )

    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    # eval
    start = time.time()
    h_loss_record, h_acc_record, l_loss_record, l_acc_record = teacher_eval(t_model, val_loader)

    logger.add_scalar('t_val/teacher_high_pressure_loss', h_loss_record.avg, epoch+1)
    logger.add_scalar('t_val/teacher_high_pressure_acc', h_acc_record.avg, epoch+1)
    logger.add_scalar('t_val/teacher_low_pressure_loss', l_loss_record.avg, epoch+1)
    logger.add_scalar('t_val/teacher_low_pressure_acc', l_acc_record.avg, epoch+1)

    run_time = time.time() - start
    msg = 'teacher val Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} hp_acc:{:.2f} lp loss:{:.3f} lp acc:{:.2f}'.format(
        epoch+1, args.t_epoch, run_time, h_loss_record.avg, h_acc_record.avg, l_loss_record.avg, l_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    if h_acc_record.avg > best_high_acc:
        state_dict = dict(epoch=epoch + 1, state_dict=t_model.high_pressure_encoder.state_dict(), acc=h_acc_record.avg)
        os.makedirs(osp.dirname(high_pressure_state_dict), exist_ok=True)
        torch.save(state_dict, high_pressure_state_dict)
        best_high_acc = h_acc_record.avg

    if l_acc_record.avg > best_low_acc:
        state_dict = dict(epoch=epoch + 1, state_dict=t_model.low_pressure_encoder.state_dict(), acc=l_acc_record.avg)
        os.makedirs(osp.dirname(low_pressure_state_dict), exist_ok=True)
        torch.save(state_dict, low_pressure_state_dict)
        best_low_acc = l_acc_record.avg

    t_high_scheduler.step()
    t_low_scheduler.step()


print(f"Teacher high:{best_high_acc} low:{best_low_acc}")


backbone_weights = torch.load(ckpt_path)['model']
high_encoder_weights = torch.load(high_pressure_state_dict)['state_dict']
low_encoder_weights = torch.load(low_pressure_state_dict)['state_dict']
t_model.backbone.load_state_dict(backbone_weights)
t_model.high_pressure_encoder.load_state_dict(high_encoder_weights)
t_model.low_pressure_encoder.load_state_dict(low_encoder_weights)
# ----------------  start distillation ! -------------------

s_model = model_dict[args.s_arch](num_classes=100)
s_model = wrapper(module=s_model, cfg=args).cuda()
s_optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
s_scheduler = MultiStepLR(s_optimizer, milestones=args.milestones, gamma=args.gamma)

best_acc = -1
for epoch in range(args.epoch):
    s_model.train()
    s_high_pressure_loss_record = AverageMeter()
    s_low__pressure_loss_record = AverageMeter()
    s_logits_loss_record = AverageMeter()
    s_acc_record = AverageMeter()

    start = time.time()
    for img, target in train_loader:
        img = img.cuda()
        target = target.cuda()

        s_optimizer.zero_grad()
        with torch.no_grad():
            t_out, t_high_pressure_encoder_out, t_low_pressure_encoder_out, _ = t_model.forward(img, bb_grad=False, decoder_train=False)

        s_out, s_high_pressure_encoder_out, s_low_pressure_encoder_out, _ = s_model.forward(img, bb_grad=True, decoder_train=False)

        logits_loss = F.kl_div(
            F.log_softmax(s_out / 4.0, dim=1),
            F.softmax(t_out / 4.0, dim=1),
            reduction='batchmean'
        ) * 4.0 * 4.0 * (1 - args.alpha) + F.cross_entropy(s_out, target) * args.alpha

        high_loss = F.kl_div(
            F.log_softmax(s_high_pressure_encoder_out / 2.0, dim=1),
            F.softmax(t_high_pressure_encoder_out / 2.0, dim=1),
            reduction='batchmean'
        ) * 2.0 * 2.0

        low_loss = F.kl_div(
            F.log_softmax(s_low_pressure_encoder_out / 8.0, dim=1),
            F.softmax(t_low_pressure_encoder_out / 8.0, dim=1),
            reduction='batchmean'
        ) * 8.0 * 8.0

        loss = logits_loss + high_loss + low_loss
        loss.backward()
        s_optimizer.step()

        s_high_pressure_loss_record.update(high_loss.item(), img.size(0))
        s_low__pressure_loss_record.update(low_loss.item(), img.size(0))
        s_logits_loss_record.update(logits_loss.item(), img.size(0))
        acc = accuracy(s_out.data, target)[0]
        s_acc_record.update(acc.item(), img.size(0))

    logger.add_scalar('s_train/s_high_loss', s_high_pressure_loss_record.avg, epoch+1)
    logger.add_scalar('s_train/s_low_loss', s_low__pressure_loss_record.avg, epoch+1)
    logger.add_scalar('s_train/s_logits_loss', s_logits_loss_record.avg, epoch+1)
    logger.add_scalar('s_train/s_acc', s_acc_record.avg, epoch+1)

    run_time = time.time() - start
    msg = 'student train Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} lp loss:{:.3f} acc:{:.2f} '.format(
        epoch + 1, args.epoch, run_time, s_high_pressure_loss_record.avg, s_low__pressure_loss_record.avg, s_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)


    # validation
    start = time.time()

    s_high_pressure_loss_record, s_logits_loss_record, s_low__pressure_loss_record, s_acc_record = student_eval(t_model, s_model, val_loader)
    logger.add_scalar('s_val/s_high_loss', s_high_pressure_loss_record.avg, epoch+1)
    logger.add_scalar('s_val/s_low_loss', s_low__pressure_loss_record.avg, epoch+1)
    logger.add_scalar('s_val/s_logits_loss', s_logits_loss_record.avg, epoch+1)
    logger.add_scalar('s_val/s_acc', s_acc_record.avg, epoch+1)
    run_time = time.time() - start

    msg = 'student val Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} lp loss:{:.3f} acc:{:.2f} '.format(
        epoch + 1, args.epoch, run_time, s_high_pressure_loss_record.avg, s_low__pressure_loss_record.avg, s_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    if s_acc_record.avg > best_acc:
        state_dict = dict(epoch=epoch+1, state_dict=s_model.state_dict(), acc=s_acc_record.avg)
        name = osp.join(exp_path, 'ckpt/student_best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
        best_acc = s_acc_record.avg

    s_scheduler.step()


print('best_acc: {:.2f}'.format(best_acc))
