import torch.nn as nn
import torch
import math
from src.t2icount.ldm.modules.attention import FeedForward


class Regressor(nn.Module):
    def __init__(self, out_dim):
        super(Regressor, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(out_dim, out_dim, 3, 1, padding=1), nn.ReLU()])
        self.img_pe_transform1 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True, groups=out_dim)
        self.self_attn1 = Attention(out_dim, 8)
        self.norm1 = nn.LayerNorm(out_dim)
        self.img_pe_transform2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True, groups=out_dim)
        self.self_attn2 = Attention(out_dim, 8)
        self.norm2 = nn.LayerNorm(out_dim)

        self.reg_head = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        B, _, H, W = x.shape
        x_pe = self.img_pe_transform1(x)
        x = x.flatten(2).transpose(1, 2)
        x_pe = x_pe.flatten(2).transpose(1, 2)
        attn_out = self.self_attn1(q=x + x_pe, k=x + x_pe, v=x)
        x = x + attn_out
        x = self.norm1(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.conv(x)
        x_pe = self.img_pe_transform2(x)
        x = x.flatten(2).transpose(1, 2)
        x_pe = x_pe.flatten(2).transpose(1, 2)
        attn_out = self.self_attn2(q=x + x_pe, k=x + x_pe, v=x)
        x = x + attn_out
        x = self.norm2(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.reg_head(x)
        return x


class Upsample(nn.Module):
    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch, sem=True):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(*[nn.Conv2d(up_in_ch, up_out_ch, 3, 1, padding=1), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv2d(cat_in_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])
        if sem:
            self.text_conv = nn.Linear(768, cat_out_ch)
            self.norm1 = nn.LayerNorm(cat_out_ch)
            self.img_txt_fusion1 = ImgTxtFusion(cat_out_ch, 8)
            self.img_txt_fusion2 = ImgTxtFusion(cat_out_ch, 8)
            self.conv3 = nn.Sequential(*[nn.Conv2d(cat_out_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])
        self.cross_attn = sem

    def forward(self, low, high, text_feat=None, simi_map=None):
        low = self.up(low)
        low = self.conv1(low)
        x = torch.cat([high, low], dim=1)
        x = self.conv2(x)
        B, _, H, W = x.shape
        if simi_map is not None:
            x = x + simi_map * low
        if self.cross_attn:
            text_feat = self.text_conv(text_feat)
            text_feat = text_feat + self.norm1(text_feat)  # B, C
            text_feat, x = self.img_txt_fusion1(x, text_feat)
            x = self.conv3(x)
            text_feat, x = self.img_txt_fusion2(x, text_feat)
            return x, text_feat
        else:
            return x


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def _separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class ImgTxtFusion(nn.Module):
    def __init__(self, channel_num, num_head):
        super(ImgTxtFusion, self).__init__()
        self.img_pe_transform = nn.Conv2d(channel_num, channel_num, 3, 1, 1, bias=True, groups=channel_num)
        self.txt2img = Attention(channel_num, num_head)
        self.norm1 = nn.LayerNorm(channel_num)
        self.mlp = FeedForward(channel_num)
        self.norm2 = nn.LayerNorm(channel_num)
        self.img2txt = Attention(channel_num, num_head)
        self.norm3 = nn.LayerNorm(channel_num)

    def forward(self, img_feat, txt_feat):
        img_pe = self.img_pe_transform(img_feat)
        B, _, H, W = img_pe.shape
        img_pe = img_pe.flatten(2).transpose(1, 2)  # B, N, C
        img_feat = img_feat.flatten(2).transpose(1, 2)  # B, N, C
        txt_feat = txt_feat.unsqueeze(1) # B, 1, C

        attn_out = self.txt2img(q=txt_feat, k=img_feat+img_pe, v=img_feat)
        txt_feat_ = txt_feat + attn_out
        txt_feat_ = self.norm1(txt_feat_)

        mlp_out = self.mlp(txt_feat_)
        txt_feat_ = txt_feat_ + mlp_out
        txt_feat_ = self.norm2(txt_feat_)

        attn_out = self.img2txt(q=img_feat+img_pe, k=txt_feat_+txt_feat, v=txt_feat_)
        img_feat = img_feat + attn_out
        img_feat = self.norm3(img_feat)

        return txt_feat_.squeeze(1), img_feat.transpose(1, 2).reshape(B, -1, H, W)


