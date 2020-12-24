import torch
import torch.nn as nn
import torch.nn.functional as F

class wrapper(nn.Module):

    def __init__(self, module, cfg):

        super(wrapper, self).__init__()

        self.backbone = module
        feat_dim = list(module.children())[-1].in_features
        try:
            num_classes = self.backbone.fc.out_features
        except:
            try:
                num_classes = self.backbone.classifier.out_features
            except:
                num_classes = self.backbone.linear.out_features

        high_pressure_nodes = cfg.encoder[0]  # 64
        low_pressure_nodes = cfg.encoder[1]   # 256

        # high-pressure tube / low-dim encoder
        self.high_pressure_encoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, high_pressure_nodes)
            )
        # high-pressure tube / low-dim decoder
        self.high_pressure_decoder = nn.Sequential(
            nn.Linear(high_pressure_nodes, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes)
            )
        # low-pressure tube / high-dim encoder
        self.low_pressure_encoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, low_pressure_nodes)
        )
        # low-pressure tube / high-dim encoder
        self.low_pressure_decoder = nn.Sequential(
            nn.Linear(low_pressure_nodes, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x, bb_grad=True, output_decoder=False, output_encoder=False):
        feats, out = self.backbone(x, is_feat=True)
        feat = feats[-1].view(feats[-1].size(0), -1)

        high_pressure_encoder_out = self.high_pressure_encoder(feat)  # [b, 64]
        low_pressure_encoder_out = self.low_pressure_encoder(feat)  # [b, 256]

        if not bb_grad:
            feat = feat.detach()

        if output_decoder:
            high_pressure_decoder_out = self.high_pressure_decoder(high_pressure_encoder_out)
            low_pressure_decoder_out = self.low_pressure_decoder(low_pressure_encoder_out)
            if not output_encoder:
                return out, high_pressure_decoder_out, low_pressure_decoder_out, feat
            else:
                return out, high_pressure_encoder_out, high_pressure_decoder_out, low_pressure_encoder_out,  low_pressure_decoder_out, feat

        return out, high_pressure_encoder_out, low_pressure_encoder_out, feat
