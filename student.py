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
from distiller_zoo import DistillationStructure
from utils import AverageMeter, accuracy, student_eval_with_teacher
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


parser = argparse.ArgumentParser(description='train student network.')
parser.add_argument('--root', type=str, default='/data/wyx/datasets/cifar100')
parser.add_argument('--num_class', type=int, default=100)

parser.add_argument('--kd_func', type=str, required=True, choices=['kd', 'hint', 'attention', 'similarity', 'correlation', 'vid', 'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound', 'factor', 'nst'])

parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])
parser.add_argument('--kd_weight', type=float, default=1.0)
parser.add_argument('--ce_weight', type=float, default=1.0)
parser.add_argument('--div_weight', type=float, default=0)

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

parser.add_argument('--T', type=float, default=4.0)  # temperature
parser.add_argument('--low_T', type=float, default=2.0)  # temperature
parser.add_argument('--high_T', type=float, default=8.0)  # temperature

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=100)

# NCE distillation
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
# hint layer
parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

if args.s_arch in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
    args.lr = 0.01

t_arch = args.t_arch
exp_name = f'{args.kd_func}_mpd_T_{t_arch}_S_{args.s_arch}'
exp_path = './experiments/{}/{}'.format(exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(exp_path, exist_ok=True)

print(f"{args.kd_func} Teacher:{args.t_arch} => Student:{args.s_arch} [{exp_path}]")

logger = SummaryWriter(osp.join(exp_path, 'events'), flush_secs=10)


if args.kd_func in ['crd']:
    train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(root=args.root, batch_size=args.batch_size, num_workers=4, k=args.nce_k, mode=args.mode)
else:
    train_loader, val_loader, n_data = get_cifar100_dataloaders(root=args.root, batch_size=args.batch_size, num_workers=4, is_instance=True)
args.n_data = n_data


# student model definition
s_model = model_dict[args.s_arch](num_classes=100)
s_model = wrapper(module=s_model, cfg=args).cuda()

# teacher model loads checkpoint
ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
ak_state_dict = osp.join("experiments", "cifar100", "wrapper_teacher", args.t_arch, "teacher_high_best.pth")
dk_state_dict = osp.join("experiments", "cifar100", "wrapper_teacher", args.t_arch, "teacher_low_best.pth")

t_model = model_dict[t_arch](num_classes=100).cuda()
t_model = wrapper(module=t_model, cfg=args).cuda()

backbone_weights = torch.load(ckpt_path)['model']
ak_encoder_weights = torch.load(ak_state_dict)['encoder_state_dict']
dk_encoder_weights = torch.load(dk_state_dict)['encoder_state_dict']
ak_decoder_weights = torch.load(ak_state_dict)['decoder_state_dict']
dk_decoder_weights = torch.load(dk_state_dict)['decoder_state_dict']

t_model.backbone.load_state_dict(backbone_weights)
t_model.ak_encoder.load_state_dict(ak_encoder_weights)
t_model.dk_encoder.load_state_dict(dk_encoder_weights)
t_model.ak_decoder.load_state_dict(ak_decoder_weights)
t_model.dk_decoder.load_state_dict(dk_decoder_weights)
t_model.eval()

# ----------------  start distillation ! -------------------
print("-------------start distillation ! -------------")

# construct kd loss and optimizer

distill_struct = DistillationStructure(args, t_model, s_model)
criterion_list, module_list, trainable_list = distill_struct.get_criteon_kd(train_loader, logger)

criterion_div, criterion_kd = criterion_list  # distill loss func

s_optimizer = optim.SGD(
    chain(s_model.parameters(), trainable_list.parameters()),
    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
s_scheduler = MultiStepLR(s_optimizer, args.milestones, args.gamma)

best_acc = -1
for epoch in range(args.epoch):
    s_model.train()
    t_model.eval()
    for module in module_list:
        module.train()
    if args.kd_func == 'abound':
        module_list[1].eval()
    elif args.kd_func == 'factor':
        module_list[2].eval()


    s_ak_loss_record = AverageMeter()
    s_dk_loss_record = AverageMeter()
    s_logits_loss_record = AverageMeter()
    s_acc_record = AverageMeter()

    start = time.time()
    for data in train_loader:
        if args.kd_func in ['crd']:
            img, target, index, contrast_idx = data
            contrast_idx = contrast_idx.cuda()
        else:
            img, target, index = data

        img = img.float().cuda()
        target = target.cuda()
        index = index.cuda()

        s_optimizer.zero_grad()

        preact = False
        if args.kd_func in ['abound']:
            preact = True
        with torch.no_grad():
            t_out, t_ak_encoder_out, t_dk_encoder_out, (feat_t, feat_ts) = t_model.forward(
                img, bb_grad=False, output_decoder=False, output_encoder=True, is_feat=True, preact=preact)

        s_out, s_ak_encoder_out, s_dk_encoder_out, (feat_s, feat_ss) = s_model.forward(
            img, bb_grad=True, output_decoder=False, output_encoder=True)

        # cls loss
        loss_cls = F.cross_entropy(s_out, target)
        loss_div = criterion_div(s_out, t_out)

        if args.kd_func == 'kd':
            loss_kd = 0

        elif args.kd_func == 'hint':
            f_s = module_list[0](feat_ss[args.hint_layer])
            f_t = feat_ts[args.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)

        elif args.kd_func == 'crd':
            f_s = feat_ss[-1]
            f_t = feat_ts[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)

        elif args.kd_func == 'attention':
            g_s = feat_ss[1:-1]
            g_t = feat_ts[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)

        elif args.kd_func == 'nst':
            g_s = feat_ss[1:-1]
            g_t = feat_ts[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)

        elif args.kd_func == 'similarity':
            g_s = [feat_ss[-2]]
            g_t = [feat_ts[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)

        elif args.kd_func == 'rkd':
            f_s = feat_ss[-1]
            f_t = feat_ts[-1]
            loss_kd = criterion_kd(f_s, f_t)

        elif args.kd_func == 'pkt':
            f_s = feat_ss[-1]
            f_t = feat_ts[-1]
            loss_kd = criterion_kd(f_s, f_t)

        elif args.kd_func == 'kdsvd':
            g_s = feat_ss[1:-1]
            g_t = feat_ts[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)

        elif args.kd_func == 'correlation':
            f_s = module_list[0](feat_ss[-1])
            f_t = module_list[1](feat_ts[-1])
            loss_kd = criterion_kd(f_s, f_t)

        elif args.kd_func == 'vid' or 'afd':
            g_s = feat_ss[1:-1]
            g_t = feat_ts[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)

        elif args.kd_func == 'abound':
            # can also add loss to this stage
            loss_kd = 0

        elif args.kd_func == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif args.kd_func == 'factor':
            factor_s = module_list[0](feat_ss[-2])
            factor_t = module_list[1](feat_ts[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)

        else:
            raise NotImplementedError(args.kd_func)


        distill_loss = args.ce_weight * loss_cls + args.kd_weight * loss_kd + args.div_weight * loss_div

        dk_encoder_loss = F.kl_div(
            F.log_softmax(s_ak_encoder_out / args.low_T, dim=1),
            F.softmax(t_ak_encoder_out / args.low_T, dim=1),
            reduction='batchmean'
        ) * args.low_T * args.low_T

        ak_encoder_loss = F.kl_div(
            F.log_softmax(s_dk_encoder_out / args.high_T, dim=1),
            F.softmax(t_dk_encoder_out / args.high_T, dim=1),
            reduction='batchmean'
        ) * args.high_T * args.high_T

        loss = distill_loss + dk_encoder_loss + ak_encoder_loss
        loss.backward()
        s_optimizer.step()

        s_ak_loss_record.update(dk_encoder_loss.item(), img.size(0))
        s_dk_loss_record.update(ak_encoder_loss.item(), img.size(0))
        s_logits_loss_record.update(distill_loss.item(), img.size(0))
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

    s_ak_loss_record, s_logits_loss_record, s_dk_loss_record, s_acc_record = student_eval_with_teacher(
        t_model, s_model, val_loader, args)

    logger.add_scalar('s_val/s_ak_loss', s_ak_loss_record.avg, epoch + 1)
    logger.add_scalar('s_val/s_dk_loss', s_dk_loss_record.avg, epoch + 1)
    logger.add_scalar('s_val/s_logits_loss', s_logits_loss_record.avg, epoch + 1)
    logger.add_scalar('s_val/s_acc', s_acc_record.avg, epoch + 1)
    run_time = time.time() - start

    msg = 'student val Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} lp loss:{:.3f} acc:{:.2f} '.format(
        epoch + 1, args.epoch, run_time, s_ak_loss_record.avg,
        s_dk_loss_record.avg, s_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    if s_acc_record.avg > best_acc:
        state_dict = dict(epoch=epoch + 1, state_dict=s_model.state_dict(), acc=s_acc_record.avg)
        name = osp.join(exp_path, 'ckpt/kd_mas_vgg13_vgg8.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
        best_acc = s_acc_record.avg

    s_scheduler.step()

print('student_best_acc: {:.2f}'.format(best_acc))
print()