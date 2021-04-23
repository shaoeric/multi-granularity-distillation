import torch
import torch.nn as nn
import torch.nn.functional as F

class wrapper(nn.Module):

    def __init__(self, module, cfg):

        super(wrapper, self).__init__()

        self.backbone = module
        feat_dim = list(module.children())[-1].in_features
        num_classes = cfg.num_class

        dim_ak = cfg.encoder[0]  # 64
        dim_dk = cfg.encoder[1]   # 256

        # high-pressure tube / low-dim encoder
        self.ak_encoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim_ak)
            )
        # high-pressure tube / low-dim decoder
        self.ak_decoder = nn.Sequential(
            nn.Linear(dim_ak, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes)
            )
        # low-pressure tube / high-dim encoder
        self.dk_encoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim_dk)
        )
        # low-pressure tube / high-dim encoder
        self.dk_decoder = nn.Sequential(
            nn.Linear(dim_dk, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x, bb_grad=True, output_decoder=False, output_encoder=False, is_feat=True, preact=False):
        feats, out = self.backbone(x, is_feat=is_feat, preact=preact)
        feat = feats[-1].view(feats[-1].size(0), -1)

        ak_encoder_out = self.ak_encoder(feat)  # [b, 64]
        dk_encoder_out = self.dk_encoder(feat)  # [b, 256]

        if not bb_grad:
            feat = feat.detach()

        if output_decoder:
            ak_decoder_out = self.ak_decoder(ak_encoder_out)
            dk_decoder_out = self.dk_decoder(dk_encoder_out)
            if not output_encoder:
                return out, ak_decoder_out, dk_decoder_out, (feat, feats)
            else:
                return out, ak_encoder_out, ak_decoder_out, dk_encoder_out,  dk_decoder_out, (feat, feats)

        return out, ak_encoder_out, dk_encoder_out, (feat, feats)
