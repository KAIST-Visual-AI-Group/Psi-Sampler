from dataclasses import dataclass
from abc import ABC

from src.utils import ignore_kwargs


__MCMC_SCHEDULER__ = dict()

def register_mcmc_scheduler(name):
    def decorator(cls):
        __MCMC_SCHEDULER__[name] = cls
        return cls
    return decorator

def get_mcmc_scheduler(name):
    if name not in __MCMC_SCHEDULER__:
        raise ValueError(f"MCMC scheduler {name} not found. Available schedulers: {list(__MCMC_SCHEDULER__.keys())}")
    return __MCMC_SCHEDULER__[name]


class MCMCScheduler(ABC):
    @ignore_kwargs
    @dataclass
    class Config():
        burn_in: int = 0
        num_mcmc_steps: int = 100

    def __init__(self, CFG):
        pass

@register_mcmc_scheduler("uniform")
class UniformMCMCScheduler(MCMCScheduler):
    @ignore_kwargs
    @dataclass
    class Config(MCMCScheduler.Config):
        pass

    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)

    def __call__(self, iter):
        return 1.0, True