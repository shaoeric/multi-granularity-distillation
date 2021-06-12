# 为了做学生的CAM图，需要对学生的多粒度做训练，此代码是不经过蒸馏的训练。蒸馏的学生训练直接参考student.py即可

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
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from data.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from itertools import chain
from tensorboardX import SummaryWriter
from utils import AverageMeter, accuracy, student_eval_with_target
from wrapper import wrapper
from models import model_dict


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='CAM no distillation MAG student')
parser.add_argument('--root', type=str, default='/data/wyx/datasets/cifar100')
parser.add_argument('--num_class', type=int, default=100)

parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])

parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])

parser.add_argument('--s-arch', type=str)  # student architecture

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

if args.s_arch in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
    args.lr = 0.01


exp_name = f'CAM/nodistill_mag_S_{args.s_arch}'
exp_path = './experiments/{}'.format(exp_name)
os.makedirs(exp_path, exist_ok=True)

print(f"CAM no distillation MAG Student:{args.s_arch} [{exp_path}]")

logger = SummaryWriter(osp.join(exp_path, 'events'), flush_secs=10)


train_loader, val_loader, n_data = get_cifar100_dataloaders(root=args.root, batch_size=args.batch_size, num_workers=4, is_instance=True)
args.n_data = n_data


# student model definition
s_model = model_dict[args.s_arch](num_classes=100)
s_model = wrapper(module=s_model, cfg=args).cuda()

# ----------------  start distillation ! -------------------
print("-------------start distillation ! -------------")

# construct kd loss and optimizer

s_optimizer = optim.SGD(
    chain(s_model.parameters()),
    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
s_scheduler = MultiStepLR(s_optimizer, args.milestones, args.gamma)

best_acc = -1
best_ak_acc = -1
best_dk_acc = -1
for epoch in range(args.epoch):
    s_model.train()

    s_ak_loss_record = AverageMeter()
    s_dk_loss_record = AverageMeter()
    s_logits_loss_record = AverageMeter()
    s_acc_record = AverageMeter()

    start = time.time()
    for data in train_loader:
        img, target, index = data
        img = img.float().cuda()
        target = target.cuda()
        index = index.cuda()

        s_optimizer.zero_grad()

        s_out, s_ak_decoder_out, s_dk_decoder_out, (feat_s, feat_ss) = s_model.forward(
            img, bb_grad=True, output_decoder=True, output_encoder=False)

        # cls loss
        loss_cls = F.cross_entropy(s_out, target)

        dk_decoder_loss = F.cross_entropy(s_dk_decoder_out, target)

        ak_decoder_loss = F.cross_entropy(s_ak_decoder_out, target)

        loss = loss_cls + dk_decoder_loss + ak_decoder_loss
        loss.backward()
        s_optimizer.step()

        s_ak_loss_record.update(dk_decoder_loss.item(), img.size(0))
        s_dk_loss_record.update(ak_decoder_loss.item(), img.size(0))
        s_logits_loss_record.update(loss_cls.item(), img.size(0))
        acc = accuracy(s_out.data, target)[0]
        s_acc_record.update(acc.item(), img.size(0))

    logger.add_scalar('s_train/s_ak_loss', s_ak_loss_record.avg, epoch + 1)
    logger.add_scalar('s_train/s_dk_loss', s_dk_loss_record.avg, epoch + 1)
    logger.add_scalar('s_train/s_logits_loss', s_logits_loss_record.avg, epoch + 1)
    logger.add_scalar('s_train/s_acc', s_acc_record.avg, epoch + 1)

    run_time = time.time() - start
    msg = 'student train Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} lp loss:{:.3f} acc:{:.2f} '.format(
        epoch + 1, args.epoch, run_time, s_ak_loss_record.avg,
        s_dk_loss_record.avg, s_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    # validation
    start = time.time()

    s_nk_acc_record, s_ak_acc_record, s_dk_acc_record = student_eval_with_target(s_model, val_loader, args)

    logger.add_scalar('s_val/s_ak_acc', s_ak_acc_record.avg, epoch + 1)
    logger.add_scalar('s_val/s_dk_acc', s_dk_acc_record.avg, epoch + 1)
    logger.add_scalar('s_val/s_logits_acc', s_nk_acc_record.avg, epoch + 1)

    run_time = time.time() - start

    msg = 'student val Epoch:{:03d}/{:03d}\truntime:{:.3f}\t ak_acc:{:.2f} nk_acc:{:.2f} dk_acc:{:.2f}'.format(
        epoch + 1, args.epoch, run_time, s_ak_acc_record.avg,
        s_nk_acc_record.avg, s_dk_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    if s_nk_acc_record.avg > best_acc:
        state_dict = dict(epoch=epoch + 1, state_dict=s_model.state_dict(), acc=s_nk_acc_record.avg)
        name = osp.join(exp_path, 'ckpt/best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
        best_acc = s_nk_acc_record.avg
        best_ak_acc = s_ak_acc_record.avg
        best_dk_acc = s_dk_acc_record.avg

    s_scheduler.step()

print('student_best_acc: {:.2f} best_ak_acc:{:.2f} best_dk_acc:{:.2f}'.format(best_acc, best_ak_acc, best_dk_acc))
print()