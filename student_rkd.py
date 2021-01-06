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
from data.cifar100 import get_cifar100_dataloaders
from distiller_zoo import RKDLoss
from itertools import chain
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy, student_eval
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


parser = argparse.ArgumentParser(description='train RKD student network.')
parser.add_argument('--root', type=str, default='/data/wyx/datasets/cifar100')

parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])

parser.add_argument('--kd_weight', type=float, default=1.0)
parser.add_argument('--ce_weight', type=float, default=1.0)

parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])

parser.add_argument('--s-arch', type=str)  # student architecture
parser.add_argument('--t-arch', type=str)  # teacher architecture
parser.add_argument('--t-path', type=str)  # teacher checkpoint path

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
exp_name = f'rkd_mpd_T_{t_arch}_S_{args.s_arch}'
exp_path = './experiments/{}/{}'.format(exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(exp_path, exist_ok=True)

print(f"RKD Teacher:{args.t_arch} => Student:{args.s_arch} [{exp_path}]")

logger = SummaryWriter(osp.join(exp_path, 'events'), flush_secs=10)

train_loader, val_loader, n_data = get_cifar100_dataloaders(root=args.root, batch_size=args.batch_size, num_workers=4, is_instance=True)

# student model definition
s_model = model_dict[args.s_arch](num_classes=100)
s_model = wrapper(module=s_model, cfg=args).cuda()

# teacher model loads checkpoint
ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
high_pressure_state_dict = osp.join("experiments", "cifar100", "wrapper_teacher", args.t_arch, "teacher_high_best.pth")
low_pressure_state_dict = osp.join("experiments", "cifar100", "wrapper_teacher", args.t_arch, "teacher_low_best.pth")

t_model = model_dict[t_arch](num_classes=100).cuda()
t_model = wrapper(module=t_model, cfg=args).cuda()

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
t_model.eval()

# ----------------  start distillation ! -------------------
print("-------------start distillation ! -------------")

criteon_kd = RKDLoss().cuda()

s_optimizer = optim.SGD(
    chain(s_model.parameters()),
    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
s_scheduler = MultiStepLR(s_optimizer, args.milestones, args.gamma)

best_acc = -1
for epoch in range(args.epoch):
    s_model.train()
    t_model.eval()

    s_high_pressure_loss_record = AverageMeter()
    s_low__pressure_loss_record = AverageMeter()
    s_logits_loss_record = AverageMeter()
    s_acc_record = AverageMeter()

    start = time.time()
    for img, target, index in train_loader:
        img = img.float().cuda()
        target = target.cuda()
        index = index.cuda()

        s_optimizer.zero_grad()
        with torch.no_grad():
            t_out, t_high_pressure_encoder_out, t_low_pressure_encoder_out, feat_t = t_model.forward(
                img, bb_grad=False, output_decoder=False, output_encoder=True)

        s_out, s_high_pressure_encoder_out, s_low_pressure_encoder_out, feat_s = s_model.forward(
            img, bb_grad=True, output_decoder=False, output_encoder=True)

        # rkd loss
        f_s = feat_s.view(img.size(0), -1)
        f_t = feat_t.view(img.size(0), -1)

        logits_loss = args.ce_weight * F.cross_entropy(s_out, target) + args.kd_weight *  criteon_kd(s_out, t_out)

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
        t_model, s_model, val_loader, args)
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
