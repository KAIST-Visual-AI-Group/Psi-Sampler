from dataclasses import dataclass, field

import torch
from torch import Tensor


__SCHEDULER__ = dict()

def register_scheduler(name):
    def decorator(cls):
        __SCHEDULER__[name] = cls
        return cls
    return decorator

def get_scheduler(name):
    if name not in __SCHEDULER__:
        raise ValueError(f"Scheduler {name} not found. Available schedulers: {list(__SCHEDULER__.keys())}")
    return __SCHEDULER__[name]


@dataclass
class SchedulerOutput:
    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})

@register_scheduler("linear")
class LinearScheduler():
    def __init__(self):
        pass

    def __call__(self, t):
        return SchedulerOutput(alpha_t = 1 - t, 
                               sigma_t = t, 
                               d_alpha_t = -torch.ones_like(t), 
                               d_sigma_t = torch.ones_like(t))

    def snr_inverse(self, snr):
        return 1.0 / (1.0 + snr)
    
@register_scheduler("vp")
class VPScheduler():
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t):
        b = self.beta_min
        B = self.beta_max

        T = 0.5 * t ** 2 * (B - b) + t * b
        dT = t * (B - b) + b
        return SchedulerOutput(
            alpha_t = torch.exp(-0.5 * T),
            sigma_t = torch.sqrt(1 - torch.exp(-T)),
            d_alpha_t = -0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T)),
        )
    
    def snr_inverse(self, snr):
        T = -torch.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = (-b + torch.sqrt(b**2 + 2 * (B-b) * T)) / (B - b)
        return t