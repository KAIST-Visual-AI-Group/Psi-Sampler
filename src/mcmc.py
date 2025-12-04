import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial
from tqdm import tqdm
from PIL import Image

import torch

from src.utils import ignore_kwargs, tensor2PIL
from src.mcmc_scheduler import get_mcmc_scheduler


__MCMC__ = dict()

def register_mcmc(name):
    def decorator(cls):
        __MCMC__[name] = cls
        return cls
    return decorator

def get_mcmc(name):
    if name not in __MCMC__:
        raise ValueError(f"MCMC {name} not found. Available MCMCs: {list(__MCMC__.keys())}")
    return __MCMC__[name]


__CALL_FUNCTIONS__ = dict()

def register_call_function(name):
    def decorator(func):
        __CALL_FUNCTIONS__[name] = func
        return func
    return decorator

def get_call_function(name):
    if name not in __CALL_FUNCTIONS__:
        raise ValueError(f"Call function {name} not found. Available functions: {list(__CALL_FUNCTIONS__.keys())}")
    return __CALL_FUNCTIONS__[name]


@register_call_function("all")
def get_all_samples(samples, rewards, **kwargs):
    true_max_idx = torch.argmax(rewards, dim=0)
    best_sample = samples[true_max_idx:true_max_idx + 1]
    true_max_reward = rewards[true_max_idx]
    return samples, best_sample, true_max_reward

@register_call_function("thinning")
def get_max_reward_samples(samples, rewards, **kwargs):    
    num_particles = kwargs["num_particles"]
    indices = torch.linspace(len(samples) - 1, 0, steps=num_particles).long()
    selected_samples = samples[indices]
    
    max_idx = torch.argmax(rewards[indices], dim=0)
    best_sample = selected_samples[max_idx:max_idx + 1]
    true_max_reward = rewards[indices][max_idx]
    return selected_samples, best_sample, true_max_reward

@register_call_function("chains_thinning")
def get_chains_thinning_samples(samples, rewards, **kwargs):
    num_particles = kwargs["num_particles"]
    num_mcmc_steps = kwargs["num_mcmc_steps"]
    num_chains = kwargs["num_chains"]
    base_cnt = num_particles // num_chains
    remainder = num_particles % num_chains   
    wanted_counts = [base_cnt + (1 if i < remainder else 0)
                       for i in range(num_chains)]

    selected = []
    selected_rewards = []
    total_steps = num_chains * num_mcmc_steps
    assert samples.shape[0] >= total_steps, "samples length mismatch"

    for chain_idx, k in enumerate(wanted_counts):
        chain_indices = torch.arange(chain_idx,
                                     chain_idx + num_chains * num_mcmc_steps,
                                     step=num_chains, device=samples.device)

        stride = max(1, chain_indices.numel() // k)

        thinned_idx = chain_indices[::stride][:k]     
        selected.append(samples[thinned_idx])
        selected_rewards.append(rewards[thinned_idx])

    thinned_samples = torch.cat(selected, dim=0)
    thinned_rewards = torch.cat(selected_rewards, dim=0)
    
    max_idx = torch.argmax(thinned_rewards, dim=0)
    best_sample = thinned_samples[max_idx:max_idx + 1]
    true_max_reward = thinned_rewards[max_idx]
    return thinned_samples, best_sample, true_max_reward



class MCMC(ABC):
    @ignore_kwargs
    @dataclass
    class Config():
        model: str = "flux"
        step_size: float = 0.5
        num_mcmc_steps: int = 50
        num_particles: int = 20
        num_chains: int = 5
        burn_in : int = 50
        alpha_mcmc: float = 0.1
        custom_call_function_name: str = None
        mcmc_scheduler: str = "uniform"
        
        save_tweedies: bool = False
        misc_dir: str = None

    @abstractmethod
    def __init__(self, CFG):
        if self.cfg.custom_call_function_name is not None:
            self.custom_call_function = get_call_function(self.cfg.custom_call_function_name)
            init_particles = self.cfg.num_particles
            
            cfg_dict = {"num_particles": init_particles, "num_mcmc_steps": self.cfg.num_mcmc_steps, "num_chains": self.cfg.num_chains}
            
            self.custom_call_function = partial(self.custom_call_function, **cfg_dict)
        else:
            self.custom_call_function = lambda x, y: x
        self.mcmc_scheduler = get_mcmc_scheduler(self.cfg.mcmc_scheduler)(CFG)
        
    @abstractmethod
    def run(self, init_position, grad_reward):
        pass

    def get_stepsize_MH_bool(self, iter):
        step_scale, mh_bool = self.mcmc_scheduler(iter)
        return self.cfg.step_size * step_scale, mh_bool

    def get_grad_reward_func(self, reward_model, pipe):
        def grad_reward_func(latents, pipe, save_tweedies=False, save_dir=None, **kwargs):
            """
            NOTE: It returns reward and gradient of reward not the ones divided with alpha_mcmc.
            It is your responsibility to divide them with alpha_mcmc if you want to use them in the MCMC.
            """
            init_t = torch.tensor([1.0], device=latents.device, dtype=torch.float32).expand(latents.shape[0])
            reward_val, grad_reward, _, decoded_tweedies = pipe.get_reward_grad_vel_tweedies(latents, reward_model, init_t)
            
            cur_mcmc_step = kwargs.get("mcmc_idx", None)
            if save_tweedies and cur_mcmc_step is not None:
                if save_dir is None:
                    raise ValueError("save_dir must be provided when save_tweedies is True")
                os.makedirs(os.path.join(save_dir, "mcmc_tweedies"), exist_ok=True)

                decoded_latents = pipe.decode_latents(latents.detach())
                target_size = (256, 256)
                batch_size = decoded_tweedies.shape[0]
                collage_img = Image.new("RGB", (target_size[0] * batch_size, target_size[1] * 2))

                for i in range(batch_size):
                    decoded_latents_img = decoded_latents[i].resize(target_size)
                    decoded_tweedies_img = tensor2PIL(decoded_tweedies[i]).resize(target_size)
                    collage_img.paste(decoded_latents_img, (i * target_size[0], 0))
                    collage_img.paste(decoded_tweedies_img, (i * target_size[0], target_size[1]))
                collage_img.save(os.path.join(save_dir, "mcmc_tweedies", f"{cur_mcmc_step:05d}.png"))
            return grad_reward, reward_val

        return partial(grad_reward_func, pipe=pipe, save_tweedies=self.cfg.save_tweedies, save_dir=self.cfg.misc_dir)
    
    def __call__(self, init_position, reward_model, pipe):
        samples, rewards = self.run(init_position, self.get_grad_reward_func(reward_model, pipe))
        samples, true_best_sample, true_best_reward = self.custom_call_function(samples, rewards)
        self.true_best_sample = true_best_sample
        self.true_best_reward = true_best_reward
        self.rewards = rewards
        return samples
    
    def set_custom_call_function(self, func):
        self.custom_call_function = func

    def get_tqdm(self):
        return tqdm(range(self.cfg.num_mcmc_steps + self.cfg.burn_in), desc="MCMC", leave=False, total=self.cfg.num_mcmc_steps + self.cfg.burn_in)

@register_mcmc("mala")
class MALA(MCMC):
    @ignore_kwargs
    @dataclass
    class Config(MCMC.Config):
        pass

    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)
        super().__init__(CFG)
    
    def get_log_joint_density(self, x_prime, x, reward_val, grad_reward, step_size):
        h = step_size
        log_joint = reward_val - h / 8.0 * (torch.sum(grad_reward * grad_reward, dim=1) + torch.sum(x_prime * x_prime, dim=1))
        log_joint = log_joint + 0.5 * torch.sum((x - (1 - h / 2.0) * x_prime) * grad_reward, dim=1)
        return log_joint

    def get_log_acceptance(self, x_current, x_proposal, reward_current, reward_proposal, grad_current, grad_proposal, step_size):
        numerator = self.get_log_joint_density(x_proposal, x_current, reward_proposal / self.cfg.alpha_mcmc, grad_proposal / self.cfg.alpha_mcmc, step_size)
        denominator = self.get_log_joint_density(x_current, x_proposal, reward_current / self.cfg.alpha_mcmc, grad_current / self.cfg.alpha_mcmc, step_size)
        log_acceptance = numerator - denominator
        return log_acceptance
    
    def run(self, init_position, grad_reward):
        original_dim = init_position.shape
        init_position = init_position.reshape(original_dim[0], -1)
        samples = list()
        reward_list = list()
        acceptence_list = list()
        x_current = init_position.clone()

        tqdm_obj = self.get_tqdm()

        cur_grad, cur_reward = grad_reward(x_current.reshape(original_dim), mcmc_idx=0)
        cur_grad = cur_grad.reshape(original_dim[0], -1)
        for i in tqdm_obj:
            step_size, MH_bool = self.get_stepsize_MH_bool(i)

            x_proposal = x_current + 0.5 * step_size * (cur_grad / self.cfg.alpha_mcmc - x_current) + (step_size**0.5) * torch.randn_like(x_current)

            proposal_grad, proposal_reward = grad_reward(x_proposal.reshape(original_dim), mcmc_idx=i+1)
            proposal_grad = proposal_grad.reshape(original_dim[0], -1)
            if MH_bool:
                log_acceptance = self.get_log_acceptance(x_current, x_proposal, cur_reward, proposal_reward, cur_grad, proposal_grad, step_size)
                acceptence = torch.exp(log_acceptance)
                acceptence_clamped = torch.minimum(acceptence, torch.tensor(1.0, dtype=acceptence.dtype, device=acceptence.device))
                acceptence_list.append(acceptence_clamped)

                accepted_chain = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
                if accepted_chain.any():
                    x_current[accepted_chain] = x_proposal[accepted_chain].clone()
                    cur_grad[accepted_chain] = proposal_grad[accepted_chain].clone()
                    cur_reward[accepted_chain] = proposal_reward[accepted_chain].clone()
            else:
                x_current = x_proposal.clone()
                cur_grad = proposal_grad.clone()
                cur_reward = proposal_reward.clone()

            if i >= self.cfg.burn_in:
                samples.append(x_current.clone())
                if i > self.cfg.burn_in:
                    reward_list.append(cur_reward.clone().detach())

        _, reward = grad_reward(x_current.reshape(original_dim), mcmc_idx=i+1)
        reward_list.append(reward.clone().detach())
        self.acceptence_list = torch.cat(acceptence_list, dim=0)
        return torch.cat(samples, dim=0).reshape(len(samples)*self.cfg.num_chains, *original_dim[1:]), torch.cat(reward_list, dim=0).reshape(len(samples)*self.cfg.num_chains)

@register_mcmc("pcnl")
class pCNL(MCMC):
    @ignore_kwargs
    @dataclass
    class Config(MCMC.Config):
        rho: float = 0.9

    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)
        super().__init__(CFG)
        self.cfg.step_size = self.get_step_size(self.cfg.rho)

    @staticmethod
    def get_step_size(rho):
        return 4.0 * (1 - rho) / (1 + rho)
    
    @staticmethod
    def get_rho(step_size):
        return (1.0 - step_size / 4.0) / (1.0 + step_size / 4.0)
    
    def get_log_joint_density(self, x_prime, x, reward_val, grad_reward, step_size):
        rho = self.get_rho(step_size)
        log_joint = reward_val - step_size / 8.0 * torch.sum(grad_reward * grad_reward, dim=1)
        tmp = (step_size ** 0.5) / 2.0 * ((x - rho * x_prime) / ((1.0 - rho ** 2.0) ** 0.5))
        log_joint = log_joint + torch.sum(tmp * grad_reward, dim=1)
        return log_joint

    def get_log_acceptance(self, x_current, x_proposal, reward_current, reward_proposal, grad_current, grad_proposal, step_size):
        numerator = self.get_log_joint_density(x_proposal, x_current, reward_proposal / self.cfg.alpha_mcmc, grad_proposal / self.cfg.alpha_mcmc, step_size)
        denominator = self.get_log_joint_density(x_current, x_proposal, reward_current / self.cfg.alpha_mcmc, grad_current / self.cfg.alpha_mcmc, step_size)
        log_acceptance = numerator - denominator
        return log_acceptance
    
    def run(self, init_position, grad_reward):
        original_dim = init_position.shape
        init_position = init_position.reshape(original_dim[0], -1)
        samples = list()
        reward_list = list()
        acceptence_list = list()
        x_current = init_position.clone()

        tqdm_obj = self.get_tqdm()

        cur_grad, cur_reward = grad_reward(x_current.reshape(original_dim), mcmc_idx=0)
        cur_grad = cur_grad.reshape(original_dim[0], -1)

        for i in tqdm_obj:
            step_size, MH_bool = self.get_stepsize_MH_bool(i)
            rho = self.get_rho(step_size)
            
            x_proposal = rho * x_current + ((1 - rho ** 2) ** 0.5) * (torch.randn_like(x_current) + 0.5 * (step_size ** 0.5) * cur_grad / self.cfg.alpha_mcmc)
            proposal_grad, proposal_reward = grad_reward(x_proposal.reshape(original_dim), mcmc_idx=i+1)
            proposal_grad = proposal_grad.reshape(original_dim[0], -1)

            if MH_bool:
                log_acceptance = self.get_log_acceptance(x_current, x_proposal, cur_reward, proposal_reward, cur_grad, proposal_grad, step_size)
                acceptence = torch.exp(log_acceptance)
                acceptence_clamped = torch.minimum(acceptence, torch.tensor(1.0, dtype=acceptence.dtype, device=acceptence.device))
                acceptence_list.append(acceptence_clamped)

                accepted_chain = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
                if accepted_chain.any():
                    x_current[accepted_chain] = x_proposal[accepted_chain].clone()
                    cur_grad[accepted_chain] = proposal_grad[accepted_chain].clone()
                    cur_reward[accepted_chain] = proposal_reward[accepted_chain].clone()
            else:
                x_current = x_proposal.clone()
                cur_grad = proposal_grad.clone()
                cur_reward = proposal_reward.clone()

            if i >= self.cfg.burn_in:
                samples.append(x_current.clone().detach())
                if i > self.cfg.burn_in:
                    reward_list.append(cur_reward.clone().detach())

        _, reward = grad_reward(x_current.reshape(original_dim), mcmc_idx=i+1)
        reward_list.append(reward.clone().detach())
        self.acceptence_list = torch.cat(acceptence_list, dim=0)
        return torch.cat(samples, dim=0).reshape(-1, *original_dim[1:]), torch.cat(reward_list, dim=0)