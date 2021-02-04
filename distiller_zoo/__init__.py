from .AB import ABLoss
from .AT import Attention
from .CC import Correlation
from .FitNet import HintLoss
from .FSP import FSP
from .FT import FactorTransfer
from .KD import DistillKL
from .KDSVD import KDSVD
from .NST import NSTLoss
from .PKT import PKT
from .RKD import RKDLoss
from .SP import Similarity
from .VID import VIDLoss
from .AFD import AFDloss
from .CRD import CRDLoss
import torch
import torch.nn as nn
from models.util import ConvReg, LinearEmbed, Connector, Translator, Paraphraser
from pretrain import init


distillation_func = {
    'abound': ABLoss,
    'attention': Attention,
    'correlation': Correlation,
    'hint': HintLoss,
    'fsp': FSP,
    'factor': FactorTransfer,
    'kd': DistillKL,
    'kdsvd': KDSVD,
    'nst': NSTLoss,
    'pkt': PKT,
    'rkd': RKDLoss,
    'similarity': Similarity,
    'vid': VIDLoss,
    'afd': AFDloss,
    'crd': CRDLoss
}


class DistillationStructure:
    def __init__(self, cfg, teacher, student):
        self.cfg = cfg
        self.teacher = teacher
        self.student = student

        random_data = torch.randn(2, 3, 32, 32).cuda()
        self.feat_t, _ = self.teacher.backbone(random_data, is_feat=True)
        self.feat_s, _ = self.student.backbone(random_data, is_feat=True)

    def get_criteon_kd(self, train_loader=None, logger=None):
        """
        construct KD loss function
        :param train_loader:
        :param logger:
        :return:
        """
        criterion_kd = None
        if self.cfg.kd_func in ['factor', 'fsp'] and (train_loader is None or logger is None):
            raise ValueError

        module_list = nn.ModuleList([])
        trainable_list = nn.ModuleList([])

        if self.cfg.kd_func == 'kd':
            criterion_kd = distillation_func[self.cfg.kd_func](self.cfg.T)

        elif self.cfg.kd_func in ['attention', 'nst', 'similarity', 'rkd', 'pkt', 'kdsvd']:
            criterion_kd = distillation_func[self.cfg.kd_func]()

        elif self.cfg.kd_func == 'hint':
            criterion_kd = distillation_func[self.cfg.kd_func]()
            regress_s = ConvReg(self.feat_s[self.cfg.hint_layer].shape, self.feat_t[self.cfg.hint_layer].shape)
            module_list.append(regress_s)
            trainable_list.append(regress_s)

        elif self.cfg.kd_func == 'crd':
            self.cfg.s_dim = self.feat_s[-1].shape[1]
            self.cfg.t_dim = self.feat_t[-1].shape[1]
            criterion_kd = CRDLoss(self.cfg)
            module_list.append(criterion_kd.embed_s)
            module_list.append(criterion_kd.embed_t)
            trainable_list.append(criterion_kd.embed_t)
            trainable_list.append(criterion_kd.embed_s)

        elif self.cfg.kd_func == 'correlation':
            criterion_kd = Correlation()
            embed_s = LinearEmbed(self.feat_s[-1].shape[1], self.cfg.feat_dim)
            embed_t = LinearEmbed(self.feat_t[-1].shape[1], self.cfg.feat_dim)
            module_list.append(embed_s)
            module_list.append(embed_t)
            trainable_list.append(embed_s)
            trainable_list.append(embed_t)

        elif self.cfg.kd_func == 'vid':
            s_n = [f.shape[1] for f in self.feat_s[1:-1]]
            t_n = [f.shape[1] for f in self.feat_t[1:-1]]
            criterion_kd = nn.ModuleList(
                [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
            )
            # add this as some parameters in VIDLoss need to be updated
            trainable_list.append(criterion_kd)

        elif self.cfg.kd_func == 'abound':
            s_shapes = [f.shape for f in self.feat_s[1:-1]]
            t_shapes = [f.shape for f in self.feat_t[1:-1]]
            connector = Connector(s_shapes, t_shapes)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(connector)
            init_trainable_list.append(self.student.backbone.get_feat_modules())
            criterion_kd = ABLoss(len(self.feat_s[1:-1]))
            init(self.student.backbone, self.teacher.backbone, init_trainable_list, criterion_kd, train_loader, logger, self.cfg)
            # classification
            module_list.append(connector)

        elif self.cfg.kd_func == 'factor':
            s_shape = self.feat_s[-2].shape
            t_shape = self.feat_t[-2].shape
            paraphraser = Paraphraser(t_shape)
            translator = Translator(s_shape, t_shape)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(paraphraser)
            criterion_init = nn.MSELoss()
            init(self.student.backbone, self.teacher.backbone, init_trainable_list, criterion_init, train_loader, logger, self.cfg)
            # classification
            criterion_kd = FactorTransfer()
            module_list.append(translator)
            module_list.append(paraphraser)
            trainable_list.append(translator)

        elif self.cfg.kd_func == 'fsp':
            s_shapes = [s.shape for s in self.feat_s[:-1]]
            t_shapes = [t.shape for t in self.feat_t[:-1]]
            criterion_kd = FSP(s_shapes, t_shapes)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(self.student.get_feat_modules())
            init(self.student.backbone, self.teacher.backbone, init_trainable_list, criterion_kd, train_loader, logger, self.cfg)
            # classification training
            pass

        else:
            raise NotImplementedError

        criterion_div = DistillKL(self.cfg.T)
        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)     # other knowledge distillation loss

        if torch.cuda.is_available():
            module_list.cuda()
            criterion_list.cuda()
        return criterion_list, module_list, trainable_list





