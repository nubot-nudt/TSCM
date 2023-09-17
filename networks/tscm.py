#%%
import sys

sys.path.append('..')
from re import L
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import cirtorch.functional as LF
import math
import torch.nn.functional as F
import torch
import timm
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torchvision.models as models
from netvlad import NetVLADLoupe
from torch import Tensor
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 使用一个卷积层而不是一个线性层 -> 性能增加
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # 将卷积操作后的patch铺平
            Rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class Shen(nn.Module): #整合Vit和resnet
    def __init__(self, opt=None):
        super().__init__()
        heads = 4
        d_model = 512
        dropout = 0.1
        resnet50 = models.resnet50(pretrained=True)
        Vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.linear = nn.Linear(768, 512)
        self.linear2 = nn.Linear(1024, 512)
        featuresV = list(Vit.children())[:-1] #ViT
        featuresR = list(resnet50.children())[:-3]#Res_without_last_stage
        self.backboneVV=nn.Sequential(*featuresV)
        self.backboneV = nn.Sequential(*featuresV, ClassificationHead(), self.linear)
        self.Classification=ClassificationHead()
        self.backboneRR = nn.Sequential(*featuresR)
        self.backboneR = nn.Sequential(*featuresR, GeM(), nn.Flatten())
        self.gem=GeM()
        self.Fl=nn.Flatten()

        self.HW = Rearrange('b e h w -> b (h w) e')

        self.attn1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout=dropout)


        self.ff1 = FeedForward(d_model, dropout=dropout)
        self.ff2 = FeedForward(d_model, dropout=dropout)



        self.net_vlad = NetVLADLoupe(feature_size=512, max_samples=784, cluster_size=64,
                                     output_dim=512, gating=True, add_batch_norm=False,
                                     is_training=True)
        self.net_vlad_R = NetVLADLoupe(feature_size=256, max_samples=392, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        self.net_vlad_V = NetVLADLoupe(feature_size=256, max_samples=392, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
    def forward(self, inputs):
        #ViT branch
        outVV=self.backboneVV(inputs) #(B,S,C)
        feature_V=self.linear(outVV)
        #Res branch
        outRR=self.backboneRR(inputs)
        outR = self.gem(outRR)
        outR = self.Fl(outR) #(B,C) (1*1024)for last concatenation
        outRR=self.HW(outRR) #(B,S,C)
        feature_R=self.linear2(outRR)
        #Inter_Transformer Encoder
        feature_fuse1 = feature_V + self.attn1(feature_V, feature_R, feature_R, mask=None)
        feature_fuse1 = feature_fuse1 + self.ff1(feature_fuse1)
        feature_fuse2 = feature_R + self.attn2(feature_R, feature_V, feature_V, mask=None)
        feature_fuse2 = feature_fuse2 + self.ff2(feature_fuse2)
        feature_fuse = torch.cat((feature_fuse1, feature_fuse2), dim=-2)
        feature_cat_origin = torch.cat((feature_V, feature_R), dim=-2)
        feature_fuse = torch.cat((feature_fuse, feature_cat_origin), dim=-1)

        #descriptor from Inter_Transformer Encoder(1*512)
        feature_fuse = feature_fuse.permute(0, 2, 1)
        feature_com = feature_fuse.unsqueeze(3)
        feature_com = self.net_vlad(feature_com)

        #decriptor from Res(1*256)
        feature_R = feature_R.permute(0, 2, 1)
        feature_R = feature_R.unsqueeze(-1)
        feature_R_enhanced = self.net_vlad_R(feature_R)
        #decriptor from ViT(1*256)
        feature_V= feature_V.permute(0, 2, 1)
        feature_V= feature_V.unsqueeze(-1)
        feature_V_enhanced = self.net_vlad_V(feature_V)

        #concatenate all descriptors
        feature_com = torch.cat((feature_R_enhanced, feature_com), dim=1)
        feature_com = torch.cat((feature_com, feature_V_enhanced), dim=1)
        feature_com=torch.cat((feature_com, outR), dim=1)

        return feature_com


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'))

class Backbone(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

        self.sigma_dim = 2048
        self.mu_dim = 2048

        self.backbone = Shen()


class Stu_Backbone(nn.Module):
    def __init__(self):
        super(Stu_Backbone, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        featuresR = list(resnet50.children())[:-3]  # Res去掉最后三层

        self.gem=GeM()
        self.Fl=nn.Flatten()
        self.backboneR_Stu = nn.Sequential(*featuresR)
        self.linear0 = nn.Linear(256, 1024)

        self.cHead = ClassificationHead()
        self.net_vlad = NetVLADLoupe(feature_size=196, max_samples=768, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

        self.HW = Rearrange('b e h w -> b (h w) e')
        self.PatchEmbedding = PatchEmbedding()
    def forward(self, inputs):
        #Res branch(1*1024)
        outRR = self.backboneR_Stu(inputs)
        outR = self.gem(outRR)
        outR = self.Fl(outR)  # （B,C) for last concatenation
        #descriptor(1*1024)
        outVV=self.PatchEmbedding(inputs) #(B,S,C)
        feature_V = outVV.permute(0, 2, 1)
        feature_V = feature_V.unsqueeze(-1)
        feature_V_enhanced = self.net_vlad(feature_V)
        outV= self.linear0(feature_V_enhanced)

        #concatenation
        feature_fuse = torch.cat((outV,outR), dim=-1)


        return feature_fuse


class TeacherNet(Backbone):
    def __init__(self, opt=None):
        super().__init__()
        self.id = 'teacher'
        self.mean_head = nn.Sequential(L2Norm(dim=1))

    def forward(self, inputs):
        B, C, H, W = inputs.shape                # (B, 1, 3, 224, 224)
                                                 # inputs = inputs.view(B * L, C, H, W)     # ([B, 3, 224, 224])

        backbone_output = self.backbone(inputs)      # ([B, 2048, 1, 1])
        mu = self.mean_head(backbone_output).view(B, -1)                                           # ([B, 2048]) <= ([B, 2048, 1, 1])

        return mu, torch.zeros_like(mu)


class StudentNet(TeacherNet):
    def __init__(self, opt=None):
        super().__init__()
        self.id = 'student'
        self.var_head = nn.Sequential(nn.Linear(2048, self.sigma_dim), nn.Sigmoid())
        self.backboneS = Stu_Backbone()
    def forward(self, inputs):
        B, C, H, W = inputs.shape                # (B, 1, 3, 224, 224)
        inputs = inputs.view(B, C, H, W)         # ([B, 3, 224, 224])
        backbone_output = self.backboneS(inputs)

        mu = self.mean_head(backbone_output).view(B, -1)                                           # ([B, 2048]) <= ([B, 2048, 1, 1])
        log_sigma_sq = self.var_head(backbone_output).view(B, -1)                                  # ([B, 2048]) <= ([B, 2048, 1, 1])

        return mu, log_sigma_sq


def deliver_model(opt, id):
    if id == 'tea':
        return TeacherNet(opt)
    elif id == 'stu':
        return StudentNet(opt)


if __name__ == '__main__':
    tea = TeacherNet()
    stu = StudentNet()
    inputs = torch.rand((1, 3, 224, 224))
    outputs_tea = tea(inputs)
    outputs_stu = stu(inputs)
   # print(outputs_tea.shape)
   # print(outputs_stu.shape)
   # print(tea.state_dict())
    print(outputs_tea[0].shape, outputs_tea[1].shape)
    print(outputs_stu[0].shape, outputs_stu[1].shape)
