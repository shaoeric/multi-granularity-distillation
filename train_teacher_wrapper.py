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
parser.add_argument('--num_class', type=int, default=100)

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
    ak_loss_record = AverageMeter()
    ak_acc_record = AverageMeter()
    dk_loss_record = AverageMeter()
    dk_acc_record = AverageMeter()

    for img, label in val_loader:
        img = img.cuda()
        label = label.cuda()

        with torch.no_grad():
            out, ak_decoder_out, dk_decoder_out, _ = t_model.forward(img,
                                                                                          bb_grad=False,
                                                                                          output_decoder=True)

        loss_ak = F.kl_div(F.log_softmax(ak_decoder_out / 2.0, dim=-1),
                                      F.softmax(out / 2.0, dim=-1),
                                      reduction='batchmean') * 2.0 * 2.0 + F.cross_entropy(
            ak_decoder_out, label)
        dk_pressure = F.kl_div(F.log_softmax(dk_decoder_out / 8.0, dim=-1),
                                     F.softmax(out / 8.0, dim=-1),
                                     reduction='batchmean') * 8.0 * 8.0 + F.cross_entropy(
            dk_decoder_out, label)

        ak_acc = accuracy(ak_decoder_out.data, label)[0]
        ak_acc_record.update(ak_acc.item(), img.size(0))
        ak_loss_record.update(loss_ak.item(), img.size(0))

        dk_acc = accuracy(dk_decoder_out.data, label)[0]
        dk_acc_record.update(dk_acc.item(), img.size(0))
        dk_loss_record.update(dk_pressure.item(), img.size(0))
    return ak_loss_record, ak_acc_record, dk_loss_record, dk_acc_record


best_low_acc = 0
best_higak_acc = 0
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
t_ak_optimizer = optim.SGD(
    [{'params': t_model.ak_encoder.parameters(), 'lr': args.t_lr},
     {'params': t_model.ak_decoder.parameters(), 'lr': args.t_lr}
     ], momentum=args.momentum, weight_decay=args.weight_decay)
t_ak_scheduler = MultiStepLR(t_ak_optimizer, milestones=args.t_milestones,
                               gamma=args.gamma)

t_dk_optimizer = optim.SGD(
    [{'params': t_model.dk_encoder.parameters(), 'lr': args.t_lr},
     {'params': t_model.dk_decoder.parameters(), 'lr': args.t_lr}
     ], momentum=args.momentum, weight_decay=args.weight_decay)
t_dk_scheduler = MultiStepLR(t_dk_optimizer, milestones=args.t_milestones,
                              gamma=args.gamma)

ak_state_dict = osp.join(exp_path, 'teacher_high_best.pth')
dk_state_dict = osp.join(exp_path, 'teacher_low_best.pth')

for epoch in range(args.t_epoch):
    t_model.eval()
    ak_loss_record = AverageMeter()
    ak_acc_record = AverageMeter()
    dk_loss_record = AverageMeter()
    dk_acc_record = AverageMeter()

    start = time.time()
    # train high pressure
    for img, label in wrapper_train_loader:
        img = img.cuda()
        label = label.cuda()

        t_ak_optimizer.zero_grad()

        out, ak_decoder_out, _, _ = t_model.forward(img, bb_grad=False,
                                                               output_decoder=True)
        out = out.detach()

        loss_ak = F.kl_div(F.log_softmax(ak_decoder_out / 2.0, dim=-1),
                                      F.softmax(out / 2.0, dim=-1),
                                      reduction='batchmean') * 2.0 * 2.0 + F.cross_entropy(
            ak_decoder_out, label)
        loss_ak.backward()
        t_ak_optimizer.step()

        ak_acc = accuracy(ak_decoder_out.data, label)[0]
        ak_acc_record.update(ak_acc.item(), img.size(0))
        ak_loss_record.update(loss_ak.item(), img.size(0))

    # train low pressure
    for img, label in wrapper_train_loader:
        img = img.cuda()
        label = label.cuda()

        t_dk_optimizer.zero_grad()

        out, _, dk_decoder_out, _ = t_model.forward(img, bb_grad=False,
                                                              output_decoder=True)
        out = out.detach()

        dk_pressure = F.kl_div(F.log_softmax(dk_decoder_out / 8.0, dim=-1),
                                     F.softmax(out / 8.0, dim=-1),
                                     reduction='batchmean') * 8.0 * 8.0 + F.cross_entropy(
            dk_decoder_out, label)
        dk_pressure.backward()
        t_dk_optimizer.step()

        dk_acc = accuracy(dk_decoder_out.data, label)[0]
        dk_acc_record.update(dk_acc.item(), img.size(0))
        dk_loss_record.update(dk_pressure.item(), img.size(0))

    logger.add_scalar('t_train/teacher_ak_loss', ak_loss_record.avg, epoch + 1)
    logger.add_scalar('t_train/teacher_ak_acc', ak_acc_record.avg, epoch + 1)
    logger.add_scalar('t_train/teacher_dk_loss', dk_loss_record.avg, epoch + 1)
    logger.add_scalar('t_train/teacher_dk_acc', dk_acc_record.avg, epoch + 1)

    run_time = time.time() - start
    msg = 'teacher train Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} hp_acc:{:.2f} lp loss:{:.3f} lp_acc:{:.2f}'.format(
        epoch + 1, args.t_epoch, run_time, ak_loss_record.avg, ak_acc_record.avg,
        dk_loss_record.avg, dk_acc_record.avg
    )

    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    # eval
    start = time.time()
    ak_loss_record, ak_acc_record, dk_loss_record, dk_acc_record = teacher_eval(t_model, wrapper_test_loader)

    logger.add_scalar('t_val/teacher_ak_loss', ak_loss_record.avg, epoch + 1)
    logger.add_scalar('t_val/teacher_ak_acc', ak_acc_record.avg, epoch + 1)
    logger.add_scalar('t_val/teacher_dk_loss', dk_loss_record.avg, epoch + 1)
    logger.add_scalar('t_val/teacher_dk_acc', dk_acc_record.avg, epoch + 1)

    run_time = time.time() - start
    msg = 'teacher val Epoch:{:03d}/{:03d}\truntime:{:.3f}\t hp loss:{:.3f} hp_acc:{:.2f} lp loss:{:.3f} lp acc:{:.2f}'.format(
        epoch + 1, args.t_epoch, run_time, ak_loss_record.avg, ak_acc_record.avg,
        dk_loss_record.avg, dk_acc_record.avg
    )
    if (epoch + 1) % args.print_freq == 0:
        print(msg)

    if ak_acc_record.avg > best_higak_acc:
        state_dict = dict(epoch=epoch + 1,
                          encoder_state_dict=t_model.ak_encoder.state_dict(),
                          decoder_state_dict=t_model.ak_decoder.state_dict(),
                          acc=ak_acc_record.avg)
        os.makedirs(osp.dirname(ak_state_dict), exist_ok=True)
        torch.save(state_dict, ak_state_dict)
        best_higak_acc = ak_acc_record.avg

    if dk_acc_record.avg > best_low_acc:
        state_dict = dict(epoch=epoch + 1,
                          encoder_state_dict=t_model.dk_encoder.state_dict(),
                          decoder_state_dict=t_model.dk_decoder.state_dict(),
                          acc=dk_acc_record.avg)
        os.makedirs(osp.dirname(dk_state_dict), exist_ok=True)
        torch.save(state_dict, dk_state_dict)
        best_low_acc = dk_acc_record.avg

    t_ak_scheduler.step()
    t_dk_scheduler.step()

print(f"Teacher high:{best_higak_acc} low:{best_low_acc}")