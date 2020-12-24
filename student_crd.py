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
from data.cifar100 import get_cifar100_dataloaders_sample
from crd.criterion import CRDLoss
from itertools import chain
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy
from wrapper import wrapper

from models import model_dict

torch.backends.cudnn.benchmark = True


# python student.py --s-arch resnet20 --t-path ./experiments/teacher_resnet56_seed0/
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='train SSKD student network.')
parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])
parser.add_argument('--alpha', type=float, default=0.2)

# NCE
parser.add_argument('--nce_k', type=int, default=16384, help='number of negative samples for NCE')
parser.add_argument('--nce_t', type=float, default=0.07, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])

parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t-lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30, 45])

parser.add_argument('--s-arch', type=str)  # student architecture
parser.add_argument('--t-arch', type=str)  # teacher architecture
parser.add_argument('--t-path', type=str)  # teacher checkpoint path
parser.add_argument('--t_wrapper_train', type=str2bool, default=True)  # teacher checkpoint path
parser.add_argument('--T', type=float, default=2.0)  # temperature

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

t_arch = args.t_arch
exp_name = f'mpd_T_{t_arch}_S_{args.s_arch}'
exp_path = './experiments/{}/{}'.format(exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(exp_path, exist_ok=True)

logger = SummaryWriter(osp.join(exp_path, 'events'), flush_secs=10)

train_loader, val_loader, n_data = get_cifar100_dataloaders_sample('./data',
                                                                   batch_size=args.batch_size,
                                                                   num_workers=4,
                                                                   k=args.nce_k,
                                                                   mode=args.mode)

# teacher model loads checkpoint
ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
t_model = model_dict[t_arch](num_classes=100).cuda()
try:
    state_dict = torch.load(ckpt_path)['model']
    t_model.load_state_dict(state_dict)
except:
    state_dict = torch.load(ckpt_path)['state_dict']
    t_model.load_state_dict(state_dict)

t_model = wrapper(module=t_model, cfg=args).cuda()

# first train the teacher's multi-pressure tube
t_model.eval()
t_high_pressure_optimizer = optim.SGD(
    [{'params': t_model.high_pressure_encoder.parameters(), 'lr': args.t_lr},
     {'params': t_model.high_pressure_decoder.parameters(), 'lr': args.t_lr}
     ], momentum=args.momentum, weight_decay=args.weight_decay)
t_high_scheduler = MultiStepLR(t_high_pressure_optimizer, milestones=args.t_milestones,
                               gamma=args.gamma)

t_low_pressure_optimizer = optim.SGD(
    [{'params': t_model.low_pressure_encoder.parameters(), 'lr': args.t_lr},
     {'params': t_model.low_pressure_decoder.parameters(), 'lr': args.t_lr}
     ], momentum=args.momentum, weight_decay=args.weight_decay)
t_low_scheduler = MultiStepLR(t_low_pressure_optimizer, milestones=args.t_milestones,
                              gamma=args.gamma)

high_pressure_state_dict = osp.join(exp_path, 'ckpt/teacher_high_best.pth')
low_pressure_state_dict = osp.join(exp_path, 'ckpt/teacher_low_best.pth')

# student model definition
s_model = model_dict[args.s_arch](num_classes=100)
s_model = wrapper(module=s_model, cfg=args).cuda()

print(f"Teacher:{args.t_arch} => Student:{args.s_arch} [{exp_path}]")


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
                                                                                          output_decoder=True)

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
            t_out, t_high_pressure_encoder_out, t_low_pressure_encoder_out, _ = t_model.forward(
                img, bb_grad=False, output_decoder=False, output_encoder=True)

        s_out, s_high_pressure_encoder_out, s_low_pressure_encoder_out, _ = s_model.forward(
            img, bb_grad=True, output_decoder=False, output_encoder=True)

        logits_loss = F.cross_entropy(s_out, target)

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


# --------------------------- train teacher's wrapper ----------------------------
if args.t_wrapper_train:
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
        for img, label, _, _ in train_loader:
            img = img.cuda()
            label = label.cuda()

            t_high_pressure_optimizer.zero_grad()

            out, high_pressure_decoder_out, _, _ = t_model.forward(img, bb_grad=False,
                                                                   output_decoder=True)
            out = out.detach()

            loss_high_pressure = F.kl_div(F.log_softmax(high_pressure_decoder_out / 2.0, dim=-1),
                                          F.softmax(out / 2.0, dim=-1),
                                          reduction='batchmean') * 2.0 * 2.0 + F.cross_entropy(
                high_pressure_decoder_out, label)
            loss_high_pressure.backward()
            t_high_pressure_optimizer.step()

            h_acc = accuracy(high_pressure_decoder_out.data, label)[0]
            h_acc_record.update(h_acc.item(), img.size(0))
            h_loss_record.update(loss_high_pressure.item(), img.size(0))

        # train low pressure
        for img, label, _, _ in train_loader:
            img = img.cuda()
            label = label.cuda()

            t_low_pressure_optimizer.zero_grad()

            out, _, low_pressure_decoder_out, _ = t_model.forward(img, bb_grad=False,
                                                                  output_decoder=True)
            out = out.detach()

            loss_low_pressure = F.kl_div(F.log_softmax(low_pressure_decoder_out / 8.0, dim=-1),
                                         F.softmax(out / 8.0, dim=-1),
                                         reduction='batchmean') * 8.0 * 8.0 + F.cross_entropy(
                low_pressure_decoder_out, label)
            loss_low_pressure.backward()
            t_low_pressure_optimizer.step()

            l_acc = accuracy(low_pressure_decoder_out.data, label)[0]
            l_acc_record.update(l_acc.item(), img.size(0))
            l_loss_record.update(loss_low_pressure.item(), img.size(0))

        logger.add_scalar('t_train/teacher_high_pressure_loss', h_loss_record.avg, epoch + 1)
        logger.add_scalar('t_train/teacher_high_pressure_acc', h_acc_record.avg, epoch + 1)
        logger.add_scalar('t_train/teacher_low_pressure_loss', l_loss_record.avg, epoch + 1)
        logger.add_scalar('t_train/teacher_low_pressure_acc', l_acc_record.avg, epoch + 1)

        run_time = time.time() - start
        msg = 'teacher train Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} hp_acc:{:.2f} lp loss:{:.3f} lp_acc:{:.2f}'.format(
            epoch + 1, args.t_epoch, run_time, h_loss_record.avg, h_acc_record.avg,
            l_loss_record.avg, l_acc_record.avg
        )

        if (epoch + 1) % args.print_freq == 0:
            print(msg)

        # eval
        start = time.time()
        h_loss_record, h_acc_record, l_loss_record, l_acc_record = teacher_eval(t_model, val_loader)

        logger.add_scalar('t_val/teacher_high_pressure_loss', h_loss_record.avg, epoch + 1)
        logger.add_scalar('t_val/teacher_high_pressure_acc', h_acc_record.avg, epoch + 1)
        logger.add_scalar('t_val/teacher_low_pressure_loss', l_loss_record.avg, epoch + 1)
        logger.add_scalar('t_val/teacher_low_pressure_acc', l_acc_record.avg, epoch + 1)

        run_time = time.time() - start
        msg = 'teacher val Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} hp_acc:{:.2f} lp loss:{:.3f} lp acc:{:.2f}'.format(
            epoch + 1, args.t_epoch, run_time, h_loss_record.avg, h_acc_record.avg,
            l_loss_record.avg, l_acc_record.avg
        )
        if (epoch + 1) % args.print_freq == 0:
            print(msg)

        if h_acc_record.avg > best_high_acc:
            state_dict = dict(epoch=epoch + 1,
                              encoder_state_dict=t_model.high_pressure_encoder.state_dict(),
                              decoder_state_dict=t_model.high_pressure_decoder.state_dict(),
                              acc=h_acc_record.avg)
            os.makedirs(osp.dirname(high_pressure_state_dict), exist_ok=True)
            torch.save(state_dict, high_pressure_state_dict)
            best_high_acc = h_acc_record.avg

        if l_acc_record.avg > best_low_acc:
            state_dict = dict(epoch=epoch + 1,
                              encoder_state_dict=t_model.low_pressure_encoder.state_dict(),
                              decoder_state_dict=t_model.low_pressure_decoder.state_dict(),
                              acc=l_acc_record.avg)
            os.makedirs(osp.dirname(low_pressure_state_dict), exist_ok=True)
            torch.save(state_dict, low_pressure_state_dict)
            best_low_acc = l_acc_record.avg

        t_high_scheduler.step()
        t_low_scheduler.step()

    print(f"Teacher high:{best_high_acc} low:{best_low_acc}")

# if train teacher's wrapper
try:
    backbone_weights = torch.load(ckpt_path)['model']
    high_encoder_weights = torch.load(high_pressure_state_dict)['encoder_state_dict']
    low_encoder_weights = torch.load(low_pressure_state_dict)['encoder_state_dict']
    high_decoder_weights = torch.load(high_pressure_state_dict)['decoder_state_dict']
    low_decoder_weights = torch.load(low_pressure_state_dict)['decoder_state_dict']

    t_model.backbone.load_state_dict(backbone_weights)
    t_model.high_pressure_encoder.load_state_dict(high_encoder_weights)
    t_model.low_pressure_encoder.load_state_dict(low_encoder_weights)
    t_model.high_pressure_decoder.load_state_dict(high_decoder_weights)
    t_model.low_pressure_decoder.load_state_dict(low_decoder_weights)

# if not train teacher's wrapper, we need to copy the wrapper file to ./experiment/wrapper/{t_arch}/teacher_{high/low}_best.pth_
except:
    backbone_weights = torch.load(ckpt_path)['model']
    high_encoder_weights = \
    torch.load(osp.join("experiments", "wrapper_teacher", args.t_arch, "teacher_high_best.pth"))[
        'encoder_state_dict']
    low_encoder_weights = \
    torch.load(osp.join("experiments", "wrapper_teacher", args.t_arch, "teacher_low_best.pth"))[
        'encoder_state_dict']
    high_decoder_weights = \
    torch.load(osp.join("experiments", "wrapper_teacher", args.t_arch, "teacher_high_best.pth"))[
        'decoder_state_dict']
    low_decoder_weights = \
    torch.load(osp.join("experiments", "wrapper_teacher", args.t_arch, "teacher_low_best.pth"))[
        'decoder_state_dict']

    t_model.backbone.load_state_dict(backbone_weights)
    t_model.high_pressure_encoder.load_state_dict(high_encoder_weights)
    t_model.low_pressure_encoder.load_state_dict(low_encoder_weights)
    t_model.high_pressure_decoder.load_state_dict(high_decoder_weights)
    t_model.low_pressure_decoder.load_state_dict(low_decoder_weights)

# ----------------  start distillation ! -------------------
print("-------------start distillation ! -------------")

# config the crd structure
random_data = torch.randn(2, 3, 32, 32).cuda()
feat_t, _ = t_model.backbone(random_data, is_feat=True)
feat_s, _ = s_model.backbone(random_data, is_feat=True)
args.t_dim = feat_t[-1].shape[1]
args.s_dim = feat_s[-1].shape[1]
args.n_data = n_data
criteon_kd = CRDLoss(args).cuda()

s_optimizer = optim.SGD(
    chain(s_model.parameters(), criteon_kd.embed_t.parameters(), criteon_kd.embed_s.parameters()),
    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
s_scheduler = MultiStepLR(s_optimizer, args.milestones, args.gamma)

best_acc = -1
for epoch in range(args.epoch):
    s_model.train()
    t_model.eval()
    criteon_kd.embed_t.train()
    criteon_kd.embed_s.train()

    s_high_pressure_loss_record = AverageMeter()
    s_low__pressure_loss_record = AverageMeter()
    s_logits_loss_record = AverageMeter()
    s_acc_record = AverageMeter()

    start = time.time()
    for img, target, index, contrast_idx in train_loader:
        img = img.float().cuda()
        target = target.cuda()
        index = index.cuda()
        contrast_idx = contrast_idx.cuda()

        s_optimizer.zero_grad()
        with torch.no_grad():
            t_out, t_high_pressure_encoder_out, t_low_pressure_encoder_out, feat_t = t_model.forward(
                img, bb_grad=False, output_decoder=False, output_encoder=True)

        s_out, s_high_pressure_encoder_out, s_low_pressure_encoder_out, feat_s = s_model.forward(
            img, bb_grad=True, output_decoder=False, output_encoder=True)

        # crd loss
        f_s = feat_s.view(img.size(0), -1)
        f_t = feat_t.view(img.size(0), -1)

        logits_loss = F.cross_entropy(s_out, target) + 0.8 * criteon_kd(f_s, f_t, index,
                                                                        contrast_idx)

        high_encoder_loss = F.kl_div(
            F.log_softmax(s_high_pressure_encoder_out / 2.0, dim=1),
            F.softmax(t_high_pressure_encoder_out / 2.0, dim=1),
            reduction='batchmean'
        ) * 2.0 * 2.0

        low_encoder_loss = F.kl_div(
            F.log_softmax(s_low_pressure_encoder_out / 8.0, dim=1),
            F.softmax(t_low_pressure_encoder_out / 8.0, dim=1),
            reduction='batchmean'
        ) * 8.0 * 8.0

        loss = logits_loss + high_encoder_loss + low_encoder_loss
        loss.backward()
        s_optimizer.step()

        s_high_pressure_loss_record.update(high_encoder_loss.item(), img.size(0))
        s_low__pressure_loss_record.update(low_encoder_loss.item(), img.size(0))
        s_logits_loss_record.update(logits_loss.item(), img.size(0))
        acc = accuracy(s_out.data, target)[0]
        s_acc_record.update(acc.item(), img.size(0))

    logger.add_scalar('s_train/s_high_loss', s_high_pressure_loss_record.avg, epoch + 1)
    logger.add_scalar('s_train/s_low_loss', s_low__pressure_loss_record.avg, epoch + 1)
    logger.add_scalar('s_train/s_logits_loss', s_logits_loss_record.avg, epoch + 1)
    logger.add_scalar('s_train/s_acc', s_acc_record.avg, epoch + 1)

    run_time = time.time() - start
    msg = 'student train Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} lp loss:{:.3f} acc:{:.2f} '.format(
        epoch + 1, args.epoch, run_time, s_high_pressure_loss_record.avg,
        s_low__pressure_loss_record.avg, s_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    # validation
    start = time.time()

    s_high_pressure_loss_record, s_logits_loss_record, s_low__pressure_loss_record, s_acc_record = student_eval(
        t_model, s_model, val_loader)
    logger.add_scalar('s_val/s_high_loss', s_high_pressure_loss_record.avg, epoch + 1)
    logger.add_scalar('s_val/s_low_loss', s_low__pressure_loss_record.avg, epoch + 1)
    logger.add_scalar('s_val/s_logits_loss', s_logits_loss_record.avg, epoch + 1)
    logger.add_scalar('s_val/s_acc', s_acc_record.avg, epoch + 1)
    run_time = time.time() - start

    msg = 'student val Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} lp loss:{:.3f} acc:{:.2f} '.format(
        epoch + 1, args.epoch, run_time, s_high_pressure_loss_record.avg,
        s_low__pressure_loss_record.avg, s_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    if s_acc_record.avg > best_acc:
        state_dict = dict(epoch=epoch + 1, state_dict=s_model.state_dict(), acc=s_acc_record.avg)
        name = osp.join(exp_path, 'ckpt/student_best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
        best_acc = s_acc_record.avg

    s_scheduler.step()

print('best_acc: {:.2f}'.format(best_acc))
