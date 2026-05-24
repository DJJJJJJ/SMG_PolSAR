import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
import os

def seeds_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  ##  CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class SpatialSelfAttention(nn.Module):
    '''
    ----------multi-heads spatial self attention-----------
    '''

    def __init__(self, in_dim, heads, dropout_rate, down_rate):
        super(SpatialSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim * heads
        self.scale = self.head_dim ** 0.5
        self.LN = nn.LayerNorm(in_dim)

        self.query = nn.Linear(in_dim, in_dim * heads)
        self.key = nn.Linear(in_dim, in_dim * heads)
        self.value = nn.Linear(in_dim, in_dim * heads)

        self.out = nn.Linear(in_dim * heads, in_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.phy_group_conv = nn.Conv2d(1, in_dim, kernel_size=down_rate + 1,
                                        stride=down_rate, padding=1)

    def forward(self, x, psc):
        b, c, _ = x.shape
        psc = self.phy_group_conv(psc)  # [b,1,h,w] -> [b,c,h,w]
        psc = psc.contiguous().view(b, c, -1)

        #   linear transformation
        q = self.query(psc)
        # q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        #   reshape for multi-heads attention
        q = q.view(b, c, self.heads, self.head_dim // self.heads)  # [b,c,heads,h*w]
        k = k.view(b, c, self.heads, self.head_dim // self.heads)  # [b,c,heads,h*w]
        v = v.view(b, c, self.heads, self.head_dim // self.heads)  # [b,c,heads,h*w]

        #   transpose dimension for matrix multiplication
        q = q.permute(0, 2, 3, 1)  # [b,heads,h*w,c],  head_dim=h*w;
        k = k.permute(0, 2, 3, 1)  # [b,heads,h*w,c]
        v = v.permute(0, 2, 3, 1)  # [b,heads,h*w,c] [b,1,h*w,c]

        #   scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [b,heads,head_dim, head_dim]
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [b,heads,h*w,c]

        #   concatente and reshape
        out = out.permute(0, 3, 1, 2)  # [b,c,heads,h*w]
        out = out.contiguous().view(b, c, self.head_dim)

        out = self.LN(self.out(out) + x)

        return out, attn_weights


class ChannelSelfAttention(nn.Module):
    '''
          multi-heads channel self attention
    '''

    def __init__(self, in_dim, embedding_dim, heads, dropout_rate, down_rate):
        super(ChannelSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = embedding_dim * heads
        self.scale = self.head_dim ** 0.5
        self.LN = nn.LayerNorm(in_dim)

        self.query = nn.Linear(in_dim, embedding_dim * heads)
        self.key = nn.Linear(in_dim, embedding_dim * heads)
        self.value = nn.Linear(in_dim, embedding_dim * heads)
        self.out = nn.Linear(embedding_dim * heads, in_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.phy_group_conv = nn.Conv2d(1, in_dim, kernel_size=down_rate + 1,
                                        stride=down_rate, padding=1)

    def forward(self, x, psc):
        b, c, _ = x.shape

        psc = self.phy_group_conv(psc)  # [b,10,h,w] -> [b,c,h,w]
        psc = psc.contiguous().view(b, c, -1)

        #   linear transformation
        q = self.query(psc)
        # q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        #   reshape for multi-heads attention
        q = q.view(b, c, self.heads, self.head_dim // self.heads) # 
        k = k.view(b, c, self.heads, self.head_dim // self.heads)
        v = v.view(b, c, self.heads, self.head_dim // self.heads)

        #   transpose dimension for matrix multiplication
        q = q.permute(0, 2, 1, 3) 
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        #   scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [b,heads,c,c]
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        #   concatente and reshape
        out = out.transpose(1, 2).contiguous().view(b, c, self.head_dim)

        out = self.LN(self.out(out) + x)
        return out, attn_weights


class SMGNET(nn.Module):
    '''
    ------------the architecture of SMGNET-----------------
    '''

    def __init__(self, in_cha, num_classes, heads):
        super(SMGNET, self).__init__()

        self.num_classes = num_classes
        self.heads = heads

        self.backbone = BackBoneNet(in_channel=in_cha, out_channel=64, kernel_size=5,
                                    stride=1, padding=2)
        self.spatial_attention = nn.ModuleList(
            [SpatialSelfAttention(in_dim=4 * 4, heads=heads, dropout_rate=0., down_rate=16)
             for _ in range(10)]
        )
        self.channel_attention = nn.ModuleList(
            [ChannelSelfAttention(in_dim=4 * 4, embedding_dim=4 * 4, heads=heads, dropout_rate=0.,
                                  down_rate=16)
             for _ in range(10)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(640 , 256, kernel_size=4, stride=1, padding=0),  ## 1*1
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)

        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            # nn.Linear(64*16, 128),# backbone only
            nn.Linear(128, num_classes)
        )

    def forward(self, inputs):
        # ---------------------------------
        psc = inputs[:, 0:10, :, :]
        x = inputs[:, 10:, :, :]
        # ---feature extraction------
        _, _, high_level = self.backbone(x)  # [8, 64, 4, 4]
        b, c, h, w = high_level.shape
        out_x = high_level.contiguous().view(b, c, -1)  # [8, 64, 16]
        # ---------------------------------
        # 在 forward 里只生成一次，且与 token 数对齐
        b, c, h, w = high_level.shape  # h,w 是 token 网格大小
        # ---------------------------------
        z_list = []
        # ---scattering contribution inspired attention-------------
        for i in range(psc.shape[1]):
            psc_ = psc[:, i, :, :].view(b, 1, 128, 128)  # [8,1,  128, 128]
            out_kv, pasa_att = self.spatial_attention[i](out_x, psc_)
            # # out_kv= 0
            # # out_q = 0
            out_q, paca_att = self.channel_attention[i](out_x, psc_)
            z_list.append(out_q + out_kv)
            # z_list.append(out_x)
        # concatenate the results from different parts
        out = torch.stack(z_list, dim=1)  # [8, 10, 64, 4*4]
        # out = out_x.view(b, c, h, w)
        # ---conv fusion-----------------
        out = out.contiguous().view(b, num_parts * c, h, w)  # [8, 10 *64, 4, 4]

        out = self.conv2(out)

        # ---Classifier------------------
        out = out.contiguous().view(b, -1)
        # out = out_x.contiguous().view(b, -1) # backbone only
        out = self.classifier(out)
        # out = torch.squeeze(torch.squeeze(out, 2), 2)
        return out



