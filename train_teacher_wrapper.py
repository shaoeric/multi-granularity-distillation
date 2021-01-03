from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import os.path as osp
import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.datasets import CIFAR100
from models import model_dict
from tensorboardX import SummaryWriter
from utils import AverageMeter, accuracy
from wrapper import wrapper
import argparse
import time

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='train teacher wrapper')
parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--t-arch', type=str)  # teacher architecture
parser.add_argument('--t-path', type=str)  # teacher checkpoint path

parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--t-lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30, 45])

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

t_arch = args.t_arch
exp_path = f'./experiments/wrapper_teacher/{t_arch}'
os.makedirs(exp_path, exist_ok=True)

logger = SummaryWriter(osp.join(exp_path, 'events'), flush_secs=10)

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


best_low_acc = 0
best_high_acc = 0
wrapper_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])
wrapper_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

wrapper_trainset = CIFAR100('./data', train=True, transform=wrapper_transform_train)
wrapper_train_loader = DataLoader(wrapper_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                          pin_memory=False)
wrapper_testset = CIFAR100('./data', train=False, transform=wrapper_transform_test)
wrapper_test_loader = DataLoader(wrapper_testset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                          pin_memory=False)

# teacher model loads checkpoint
ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
t_model = model_dict[t_arch](num_classes=100).cuda()
state_dict = torch.load(ckpt_path)['model']
t_model.load_state_dict(state_dict)
t_model = wrapper(module=t_model, cfg=args).cuda()

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

high_pressure_state_dict = osp.join(exp_path, 'teacher_high_best.pth')
low_pressure_state_dict = osp.join(exp_path, 'teacher_low_best.pth')

for epoch in range(args.t_epoch):
    t_model.eval()
    h_loss_record = AverageMeter()
    h_acc_record = AverageMeter()
    l_loss_record = AverageMeter()
    l_acc_record = AverageMeter()

    start = time.time()
    # train high pressure
    for img, label in wrapper_train_loader:
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
    for img, label in wrapper_train_loader:
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
    h_loss_record, h_acc_record, l_loss_record, l_acc_record = teacher_eval(t_model, wrapper_test_loader)

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