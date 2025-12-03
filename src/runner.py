import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import *
import numpy as np


__RUNNER__ = dict()

def register_runner(name):
    def decorator(cls):
        __RUNNER__[name] = cls
        return cls
    return decorator

def get_runner(name):
    if name not in __RUNNER__:
        raise ValueError(f"Runner {name} not found. Available runners: {list(__RUNNER__.keys())}")
    return __RUNNER__[name]


import matplotlib.cm as cm
def get_overlay_for_save(reward_model, decoded_tweedies):
    with torch.no_grad():
        overlays = [
            reward_model.get_overlay(decoded_tweedies[i:i+1].to(torch.float32))
            for i in range(decoded_tweedies.shape[0])
        ]
        overlay = torch.cat(overlays, dim=0)

        counts = [
            reward_model.get_counts(decoded_tweedies[i:i+1].to(torch.float32))
            for i in range(decoded_tweedies.shape[0])
        ]
        counts = [
            ','.join([f"{cnt:.2f}" for cnt in count])
            for count in counts
        ]

    # overlay: (num_particles, 384, 384)
    overlay = overlay.unsqueeze(1)
    _, _, h, w = decoded_tweedies.shape
    overlay = F.interpolate(overlay, size=(h, w), mode="bilinear")

    # apply colormap to overlay
    cmap = cm.get_cmap("viridis")
    overlay_np = overlay.cpu().numpy().squeeze(1)
    colored_overlay_np = cmap(overlay_np)[..., :3]  # returns RGBA in [0, 1]

    colored_overlay = torch.from_numpy(colored_overlay_np).to(overlay.device)
    colored_overlay = colored_overlay.permute(0, 3, 1, 2)

    # overlay images on decoded_tweedies
    decoded_tweedies_overlay = decoded_tweedies * 0.3 + colored_overlay * 0.7
    decoded_tweedies_for_save = torch.cat([decoded_tweedies, decoded_tweedies_overlay], dim=0)
    reward_for_save = counts + counts
    return decoded_tweedies_for_save, reward_for_save


@register_runner("smc")
class SMC():
    @ignore_kwargs
    @dataclass
    class Config():
        num_inference_steps: int = 25
        alpha: float = 0.1
        num_particles: int = 10
        ess_threshold: float = 0.5
        smc_grad_norm: float = None
        grad_const_scale: float = None
        smc_grad_const_scale: float = None

        save_tweedies: bool = False
        save_dir: str = None
        misc_dir: str = None
        return_max_reward: bool = True
        init_weight_method: str = "reward"
        grad_scale: float = 1.0

        reward_name: str = None

    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)
        self.particle_schedule = [self.cfg.num_particles] * self.cfg.num_inference_steps

    def pretrained_over_proposal(self, latents, mu, sigma, grad_reward):
        batch_size = latents.shape[0]
        latents = latents.view(batch_size, -1)
        mu = mu.view(batch_size, -1)
        sigma = sigma.view(batch_size, -1)
        grad_reward = grad_reward.view(batch_size, -1)

        exponent = grad_reward * (-latents / self.cfg.alpha + mu / self.cfg.alpha + sigma ** 2 * grad_reward / (2 * self.cfg.alpha ** 2))
        exponent = exponent.sum(dim=-1)
        return exponent

    @torch.enable_grad()
    def run(self, pipe, time_sampler, reward_model, latents, generator=None, **kwargs):
        assert latents.shape[0] == self.particle_schedule[0], "Number of particles must match the number of latents"

        t, dt = time_sampler(torch.tensor([0], device=latents.device, dtype=torch.int32).expand(latents.shape[0]))
        cur_reward_values, cur_grad_reward, vel_pred, decoded_tweedies = pipe.get_reward_grad_vel_tweedies(latents, reward_model, t)
        if self.cfg.save_tweedies:
            bench_idx = kwargs.get("idx", None)
            dir_name = "smc_tweedies" if bench_idx is None else f"smc_tweedies_{bench_idx:05d}"
            os.makedirs(os.path.join(self.cfg.misc_dir, dir_name), exist_ok=True)
            if self.cfg.reward_name == "t2icount":
                decoded_tweedies_for_save, reward_for_save = get_overlay_for_save(reward_model, decoded_tweedies)
                save_collage(decoded_tweedies_for_save, os.path.join(self.cfg.misc_dir, dir_name, f"{0:05d}.png"), resize=(256, 256), reward_values=reward_for_save)
            else:
                save_collage(decoded_tweedies, os.path.join(self.cfg.misc_dir, dir_name, f"{0:05d}.png"), resize=(256, 256), reward_values=cur_reward_values)

        if self.cfg.init_weight_method == "original":
            cur_weights = cur_reward_values / self.cfg.alpha if pipe.cfg.mcmc == "gaussian" else torch.ones_like(cur_reward_values)
        elif self.cfg.init_weight_method == "reward":
            cur_weights = cur_reward_values / self.cfg.alpha
        else:
            raise ValueError(f"Unknown init_weight_method: {self.cfg.init_weight_method}")

        assert cur_weights.ndim == 1, "Dimension of reward values should be 1D"
        
        
        for step in tqdm(range(1, self.cfg.num_inference_steps + 1), desc="SMC Steps", leave=False):
            cur_num_particles = latents.shape[0]
            next_num_particles = self.particle_schedule[step-1]
            lse = torch.logsumexp(cur_weights - cur_weights.max(), dim=0)
            if not torch.isfinite(lse):
                lse = torch.tensor(0., device=cur_weights.device)
            normalized_weights = torch.exp(cur_weights - cur_weights.max() - lse)

            ess = 1.0 / torch.sum(normalized_weights ** 2)

            # Logging ############################################
            bench_idx = kwargs.get("idx", None)
            file_name = "values" if bench_idx is None else f"values_{bench_idx:05d}"
            
            if self.cfg.misc_dir is not None:
                with open(os.path.join(self.cfg.misc_dir, file_name+".txt"), "a") as f:
                    weight_vals = [str(tmp) for tmp in cur_weights.tolist()]
                    norm_weight_vals = [str(tmp) for tmp in normalized_weights.tolist()]
                    weight_val_str = ", ".join(weight_vals)
                    norm_weight_str = ", ".join(norm_weight_vals)
                    f.write(f"Step {step}: {ess.item()}\n")
                    f.write(f"weights: {weight_val_str}\n")
                    f.write(f"Normalized Weights: {norm_weight_str}\n")
                    f.write("#"*50+"\n\n")
                

            # Logging ############################################
                
            if ess.item() < self.cfg.ess_threshold * cur_num_particles:
                # SSP Resmampling
                ancestor_idx = torch.from_numpy(ssp(W=np.array(normalized_weights.cpu()), M=next_num_particles)).to(latents.device)

                prev_weights = torch.zeros_like(cur_weights)

            latents = latents[ancestor_idx]
            vel_pred = vel_pred[ancestor_idx]
            cur_reward_values = cur_reward_values[ancestor_idx]
            cur_grad_reward = cur_grad_reward[ancestor_idx]

            latents, mu, sigma = pipe.step(latents, t, dt, vel_pred, return_mu_sigma=True)
            latents = latents + sigma ** 2 * cur_grad_reward / self.cfg.alpha * self.cfg.grad_scale
            
            if step < self.cfg.num_inference_steps:
                t, dt = time_sampler(torch.tensor([step], device=latents.device, dtype=torch.int32).expand(latents.shape[0]))
                next_reward_values, next_grad_reward, vel_pred, decoded_tweedies = pipe.get_reward_grad_vel_tweedies(latents, reward_model, t)
                if self.cfg.save_tweedies:

                    bench_idx = kwargs.get("idx", None)
                    dir_name = "smc_tweedies" if bench_idx is None else f"smc_tweedies_{bench_idx:05d}"
                    if self.cfg.reward_name == "t2icount":
                        tweedies_for_save, reward_for_save = get_overlay_for_save(reward_model, decoded_tweedies)
                        save_collage(tweedies_for_save, os.path.join(self.cfg.misc_dir, dir_name, f"{step:05d}.png"), resize=(256, 256), reward_values=reward_for_save)
                    else:
                        save_collage(decoded_tweedies, os.path.join(self.cfg.misc_dir, dir_name, f"{step:05d}.png"), resize=(256, 256), reward_values=next_reward_values)
            else:
                next_reward_values, _, _, _ = pipe.get_reward_grad_vel_tweedies(latents, reward_model, t, return_grad=False)

            cur_weights = prev_weights + (next_reward_values - cur_reward_values) / self.cfg.alpha + self.pretrained_over_proposal(latents, mu, sigma, cur_grad_reward)

            cur_reward_values = next_reward_values.clone()
            cur_grad_reward = next_grad_reward.clone()
        
        max_weight_idx = torch.argmax(cur_weights).item() if not self.cfg.return_max_reward else torch.argmax(cur_reward_values).item()
        best_latents = latents[max_weight_idx : max_weight_idx + 1]
        best_sample_reward = cur_reward_values[max_weight_idx]
        return best_latents, best_sample_reward
