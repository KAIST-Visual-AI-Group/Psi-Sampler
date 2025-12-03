from torch import nn
from omegaconf import OmegaConf
from src.t2icount.ldm.util import instantiate_from_config
import torch
import torch.nn.functional as F
from .diff_unet import UNetWrapper
from .decoder import Upsample, Regressor


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


class Count(nn.Module):
    def __init__(self, config, sd_path, unet_config=dict()):
        super(Count, self).__init__()
        config = OmegaConf.load(f'{config}')
        sd_model = load_model_from_config(config, f"{sd_path}")
        self.vae = sd_model.first_stage_model
        self.unet = UNetWrapper(sd_model.model, **unet_config)

        self.clip = sd_model.cond_stage_model
        del self.vae.decoder
        self.set_frozen_parameters()
        self.decoder = Decoder(in_dim=[320, 640, 1280, 1280], out_dim=256)

    def set_frozen_parameters(self):
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False

    def set_train(self):
        self.unet.train()
        self.decoder.train()
        self.vae.eval()
        self.clip.eval()

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.clip.eval()
        self.decoder.eval()

    def extract_feat(self, img, prompt):
        latents = self.vae.encode(img)
        with torch.no_grad():
            class_info = self.clip.encode(prompt)
            c_crossattn = class_info

        latents = latents.mode()

        t = torch.zeros((img.shape[0],), device=img.device).long()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        return outs, c_crossattn

    def forward(self, x, prompt, prompt_attn_mask):
        outs, text_feat = self.extract_feat(x, prompt)
        img_feat, attn12, attn24, attn48 = outs
        attn12 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)(attn12)
        attn24 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(attn24)
        attn12 = (attn12 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)
        attn24 = (attn24 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)
        attn48 = (attn48 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)

        fused_attn = 0.1 * minmax_norm(attn48) + 0.3 * minmax_norm(attn24) + 0.6 * minmax_norm(attn12)
        text_feat = (text_feat * prompt_attn_mask.squeeze(3)).sum(dim=1) / prompt_attn_mask.squeeze(3).sum(dim=1) # B, 768
        den_map, sim_x2, sim_x1 = self.decoder(img_feat, text_feat)
        return den_map, sim_x2, sim_x1, fused_attn.float().detach()


class Decoder(nn.Module):
    def __init__(self, in_dim=[320, 717, 1357, 1280], out_dim=256):
        super(Decoder, self).__init__()
        self.upsample1 = Upsample(in_dim[3], in_dim[3]//2, in_dim[3]//2 + in_dim[2], in_dim[3]//2)
        self.upsample2 = Upsample(in_dim[3]//2, in_dim[3] // 4, in_dim[3]//4 + in_dim[1], in_dim[3] // 4)
        self.upsample3 = Upsample(in_dim[3]//4, out_dim, out_dim + in_dim[0], out_dim, False)

        self.regressor = Regressor(out_dim)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, outs, text_feat):
        x = outs
        feat, fused_text = self.upsample1(x[3], x[2], text_feat)
        sim_x2 = F.cosine_similarity(self.up(self.up(feat)), fused_text.unsqueeze(2).unsqueeze(3), dim=1).unsqueeze(1)
        feat, fused_text = self.upsample2(feat, x[1], text_feat,
                                          simi_map=F.interpolate(sim_x2, (24, 24), mode='bilinear', align_corners=False))
        sim_x1 = F.cosine_similarity(self.up(feat), fused_text.unsqueeze(2).unsqueeze(3), dim=1).unsqueeze(1)
        feat = self.upsample3(feat, x[0], simi_map=sim_x1)
        den = self.regressor(feat)
        return den, sim_x2, sim_x1


def minmax_norm(x):
    B, C, H, W = x.shape
    x = x.flatten(1)
    x_min = x.min(dim=1, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0]
    eps = 1e-7
    denominator = (x_max - x_min) + eps
    x_normalized = (x - x_min) / denominator
    x_normalized = x_normalized.reshape(B, C, H, W)
    return x_normalized

