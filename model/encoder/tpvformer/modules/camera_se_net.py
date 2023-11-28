import torch, torch.nn as nn
import numpy as np

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# class SELayer(nn.Module):

#     def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
#         super().__init__()
#         self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
#         self.act1 = act_layer()
#         self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
#         self.gate = gate_layer()

#     def forward(self, x, x_se):
#         x_se = self.conv_reduce(x_se)
#         x_se = self.act1(x_se)
#         x_se = self.conv_expand(x_se)
#         return x * self.gate(x_se)

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()

    def forward(self, x, x_se):
        return x * x_se

class CameraAwareSE(nn.Module):

    def __init__(
            self,
            in_channels=96,
            mid_channels=192,
            out_channles=96):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channles
        self._init_layers()

    def _init_layers(self):
        self.bn = nn.BatchNorm1d(16)
        self.context_mlp = Mlp(16, self.mid_channels, self.mid_channels)
        self.context_se = SELayer(self.mid_channels)  # NOTE: add camera-aware
        self.context_conv = nn.Conv2d(self.mid_channels,
                                      self.out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        
        if self.in_channels == self.mid_channels:
            self.reduce_conv = nn.Identity()
        else:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          self.mid_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(self.mid_channels),
                nn.ReLU(inplace=True))
    
    def init_weight(self):
        # nn.init.zeros_(self.context_se.conv_expand.weight)
        # nn.init.constant_(self.context_se.conv_expand.bias, 10.0)
        nn.init.zeros_(self.context_mlp.fc2.weight)
        nn.init.constant_(self.context_mlp.fc2.bias, 10.0)

    def forward(self, ms_img_feats, metas):
        intrins, sensor2ego = [], []
        for meta in metas:
            intrins.append(meta['intrinsic'])
            sensor2ego.append(meta['cam2ego'])
        intrins = np.asarray(intrins)
        intrins = ms_img_feats[0].new_tensor(intrins) # bs, N, 4, 4
        sensor2ego = np.asarray(sensor2ego)
        sensor2ego = ms_img_feats[0].new_tensor(sensor2ego)[..., :3, :]

        batch_size = intrins.shape[0]
        num_cams = intrins.shape[1]
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[..., 0, 0],
                        intrins[..., 1, 1],
                        intrins[..., 0, 2],
                        intrins[..., 1, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, num_cams, -1),
            ],
            -1,
        ) # bs, N, 16
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        context_se = self.context_mlp(mlp_input)[..., None, None] # bs*N, c, 1, 1
        context_se = torch.sigmoid(context_se)

        outputs = []
        for i_scale, img_feats in enumerate(ms_img_feats):
            img_feats = self.reduce_conv(img_feats.flatten(0, 1)) # bs*N, c, h, w
            img_feats = self.context_se(img_feats, context_se)
            img_feats = self.context_conv(img_feats)
            outputs.append(img_feats.unflatten(0, (batch_size, num_cams)))

        return outputs
