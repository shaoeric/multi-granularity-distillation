from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch
import cv2
from models import model_dict
from wrapper import wrapper
import os.path as osp
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='train teacher network.')
parser.add_argument('--encoder', type=int, nargs='+', default=[64, 256])
parser.add_argument('--num_class', type=int, default=100)

parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[30, 45])

parser.add_argument('--save-interval', type=int, default=40)
parser.add_argument('--t-arch', type=str, default='resnet32x4')
parser.add_argument('--t-path', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# # teacher model loads checkpoint
# # teacher model loads checkpoint
ckpt_path = osp.join('experiments','reconstruct','teacher_resnet32x4_seed0', 'ckpt/best.pth')
ak_state_dict = osp.join('experiments','reconstruct','teacher_resnet32x4_seed0', "ckpt/teacher_high_best.pth")
dk_state_dict = osp.join('experiments','reconstruct','teacher_resnet32x4_seed0', "ckpt/teacher_low_best.pth")

t_model = model_dict[args.t_arch](num_classes=100)
t_model = wrapper(module=t_model, cfg=args)

backbone_weights = torch.load(ckpt_path, map_location='cpu')['model']
ak_encoder_weights = torch.load(ak_state_dict, map_location='cpu')['encoder_state_dict']
dk_encoder_weights = torch.load(dk_state_dict, map_location='cpu')['encoder_state_dict']
ak_decoder_weights = torch.load(ak_state_dict, map_location='cpu')['decoder_state_dict']
dk_decoder_weights = torch.load(dk_state_dict, map_location='cpu')['decoder_state_dict']

t_model.backbone.load_state_dict(backbone_weights)
t_model.ak_encoder.load_state_dict(ak_encoder_weights)
t_model.dk_encoder.load_state_dict(dk_encoder_weights)
t_model.ak_decoder.load_state_dict(ak_decoder_weights)
t_model.dk_decoder.load_state_dict(dk_decoder_weights)
t_model.eval()

finalconv_name = 'layer3'

# for i, (name, module) in enumerate(t_model.named_parameters()):
#     print(i, name)
    # 102 backbone.fc.weight

# 获取特定层的feature map
# hook the feature extractor
features_blobs = []

def nk_hook_feature(module, input, output): # input是注册层的输入 output是注册层的输出
    print("hook input",input[0].shape)
    features_blobs.append(output[0].data.cpu().numpy())

# 对layer4层注册，把layer4层的输出append到features里面
t_model.backbone._modules.get(finalconv_name).register_forward_hook(nk_hook_feature) # 注册到finalconv_name,如果执行net()的时候，
                                                            # 会对注册的钩子也执行，这里就是执行了 layer4()
# 得到softmax weight,
params = list(t_model.parameters()) # 将参数变换为列表 按照weights bias 排列 池化无参数
nk_weight_softmax = np.squeeze(params[102].data.numpy()) # 提取fc softmax 层的参数 （weights，-1是bias）
ak_weight_softmax = np.squeeze(params[106].data.numpy()) # 提取fc softmax 层的参数 （weights，-1是bias）
dk_weight_softmax = np.squeeze(params[114].data.numpy()) # 提取fc softmax 层的参数 （weights，-1是bias）

# 生成CAM图的函数，完成权重和feature相乘操作，最后resize成上采样
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape # 获取feature_conv特征的尺寸
    output_cam = []
    #lass_idx为预测分值较大的类别的数字表示的数组，一张图片中有N类物体则数组中N个元素
    for idx in class_idx:
    # weight_softmax中预测为第idx类的参数w乘以feature_map(为了相乘，故reshape了map的形状)

        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w))) # 把原来的相乘再相加转化为矩阵
                                                                    # w1*c1 + w2*c2+ .. -> (w1,w2..) * (c1,c2..)^T -> (w1,w2...)*((c11,c12,c13..),(c21,c22,c23..))
        # 将feature_map的形状reshape回去
        cam = cam.reshape(h, w)
        # 归一化操作（最小的值为0，最大的为1）
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        # 转换为图片的255的数据
        cam_img = np.uint8(255 * cam_img)
        # resize 图片尺寸与输入图片一致
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam



data = np.load('val_mini.npy')[1]
data_tensor = torch.from_numpy(data).float()
data_tensor = data_tensor.unsqueeze(0)

with torch.no_grad():
    t_out, t_ak_encoder_out, t_ak_decoder_out, t_dk_encoder_out, \
    t_dk_decoder_out, (feat_t, feat_ts) = t_model.forward(
        data_tensor, bb_grad=False, output_decoder=True, output_encoder=True, is_feat=True)




def get_CAM(logit, weight_softmax):
    # 使用softmax打分
    h_x = F.softmax(logit, dim=1).data.squeeze()  # 分类分值
    # 对分类的预测类别分值排序，输出预测值和在列表中的位置
    probs, idx = h_x.sort(0, True)
    # 转换数据类型
    probs = probs.numpy()
    idx = idx.numpy()
    # # 输出与图片尺寸一致的CAM图片
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[2]])
    return CAMs

def show_heatmap(CAMs, target):
    height, width, channel = target.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_RAINBOW)
    result = (heatmap * 0.3 + target * 0.7).astype(int)

    plt.imshow(result)
    plt.show()

ak_CAMs = get_CAM(t_ak_encoder_out, ak_weight_softmax)
nk_CAMs = get_CAM(t_out, nk_weight_softmax)
dk_CAMs = get_CAM(t_dk_encoder_out, dk_weight_softmax)
# print('output CAM.jpg for the top1 prediction: %s'%classes[nk_idx[0]])
# # 将图片和CAM拼接在一起展示定位结果结果
mean = np.array([0.5071, 0.4866, 0.4409]).reshape(-1, 1, 1)
std = np.array([0.2675, 0.2565, 0.2761]).reshape(-1, 1, 1)
img = data

channel, height, width = img.shape
target = (img * std + mean)
target = np.transpose(target, (1, 2, 0))
target = (target * 255).astype(int)
# # 生成热度图
show_heatmap(ak_CAMs, target)
show_heatmap(nk_CAMs, target)
show_heatmap(dk_CAMs, target)
plt.imshow(target)
plt.show()