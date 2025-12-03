from dataclasses import dataclass

import torch

from src.utils import ignore_kwargs, retrieve_latents
from src.scheduler import get_scheduler
from src.mcmc import get_mcmc
from diffusers import FluxPipeline


class TimeSampler():
    @ignore_kwargs
    @dataclass
    class Config:
        num_inference_steps: int = 4
        time_schedule: str = "linear"

    def __init__(self, device, CFG):
        self.cfg = self.Config(**CFG)

        if self.cfg.time_schedule == "linear":
            self.times = torch.linspace(1.0, 1.0 / self.cfg.num_inference_steps, self.cfg.num_inference_steps, device=device)
        elif self.cfg.time_schedule == "nonlinear":
            x = torch.linspace(0.0, 1.0 - 1 / self.cfg.num_inference_steps, self.cfg.num_inference_steps, device=device)
            self.times = (1 - x ** 2) ** 0.5
        else:
            raise ValueError(f"Unknown time schedule {self.cfg.time_schedule}. Use 'linear' or 'nonlinear'.")
        
        self.times = torch.cat([self.times, torch.zeros(1, device=self.times.device)]).to(torch.float32)

    def __call__(self, step):
        if type(step) not in [torch.Tensor]:
            if type(step) in [int, float]:
                step = [step]
            step = torch.tensor(step, device=self.times.device)
        assert (step < self.cfg.num_inference_steps).all().item(), f"step {step} >= num inference step {self.cfg.num_inference_steps}"
        return self.times[step], (self.times[step] - self.times[step + 1]) 


class StochasticFluxPipeline():
    @ignore_kwargs
    @dataclass
    class Config:
        model: str = "schnell"
        mini_batch_size: int = 5
        grad_minibatch_size: int = 1
        num_particles: int = 1

        true_cfg_scale: float = 1.0
        guidance_scale: float = 3.5  # Schnell does not use this guidance_scale (Dev does)

        original_scheduler: str = "linear"
        new_scheduler: str = None

        sample_method: str = "sde"  # "ode" or "sde"
        diffuse_coeff_func: str = "original"
        diffusion_norm : float = 3.0

        mcmc: str = "gaussian"
        
        height: int = 512
        width: int = 512
        
        # For Triangle SMC
        num_inference_steps: int = 10

        save_vram : bool = False
        split_gpus: bool = False

    def __init__(self, device, CFG):
        self.cfg = self.Config(**CFG)
        self.device = device
        
        model_id = "black-forest-labs/FLUX.1-schnell"
        
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(self.device)

        # Save VRAM
        if self.cfg.save_vram:
            self.pipe.vae.eval().requires_grad_(False)
            self.pipe.text_encoder.eval().requires_grad_(False)
            self.pipe.text_encoder_2.eval().requires_grad_(False)
            self.pipe.transformer.eval().requires_grad_(False)

        self.original_scheduler = get_scheduler(self.cfg.original_scheduler)()

        if self.cfg.new_scheduler is not None:
            assert self.cfg.new_scheduler != self.cfg.original_scheduler, f"New scheduler {self.cfg.new_scheduler} must be different from original scheduler {self.cfg.original_scheduler}"
            self.new_scheduler = get_scheduler(self.cfg.new_scheduler)()
            self.do_scheduler_conversion = True
        else:
            self.new_scheduler = None
            self.do_scheduler_conversion = False

        if self.pipe.transformer.config.guidance_embeds:
            # Flux-Dev
            self.guidance = torch.full([1], self.cfg.guidance_scale, device=self.device, dtype=torch.float32)
        else:
            # Flux-Schnell
            self.guidance = None

        # Initial Latents Sampling Method
        if self.cfg.mcmc != "gaussian":
            self.init_sampling_method = get_mcmc(self.cfg.mcmc)(CFG)
        else:
            self.init_sampling_method = None

        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.height = None
        self.width = None

    def unload_encoder(self):
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")

    def load_encoder(self):
        self.pipe.text_encoder.to(self.device)
        self.pipe.text_encoder_2.to(self.device)

    def clear_cache(self):
        self.prompt_embeds = None
        self.pooled_peompt_embeds = None

        if self.negative_prompt_embeds is not None:
            self.negative_prompt_embeds = None
            self.negative_pooled_prompt_embeds = None

        self.text_ids = None
        self.latent_image_ids = None
        self.negative_prompt_embeds = None
        self.height = None
        self.width = None
    
    def encode_prompt(self, prompt, negative_prompt=None, prompt_2=None, negative_prompt_2=None, phrases=None):
        if self.cfg.split_gpus and self.pipe.text_encoder.device.type == "cpu":
            torch.cuda.empty_cache()
            self.pipe.text_encoder.to("cuda:0")
            self.pipe.text_encoder_2.to("cuda:0")
            
        self.do_true_cfg = self.cfg.true_cfg_scale > 1.0 and negative_prompt is not None

        self.prompt_embeds, self.pooled_peompt_embeds, self.text_ids = self.pipe.encode_prompt(
            prompt = prompt,
            prompt_2 = prompt_2,
            device = self.device)

        if self.do_true_cfg:
            self.negative_prompt_embeds, self.negative_pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt = negative_prompt,
                prompt_2 = negative_prompt_2,
                device = self.device)
        
        if phrases is not None:
            self.phrases_indices = self.get_t5_subsequence_indices(prompt, phrases)
        
        if self.cfg.split_gpus:
            self.pipe.text_encoder.cpu()
            self.pipe.text_encoder_2.cpu()
            torch.cuda.empty_cache()

    def get_t5_subsequence_indices(self, prompt, phrases_list):
        tokens = self.pipe.tokenizer_2.encode(prompt)
        
        phrases_indices = []
        for phrase in phrases_list:
            sub_tokens = self.pipe.tokenizer_2.encode(phrase)[:-1]
            # Find subsequence input_ids
            sub_len = len(sub_tokens)
            for i in range(len(tokens) - sub_len + 1):
                if tokens[i:i + sub_len] == sub_tokens:
                    phrases_indices.append([j for j in range(i, i + sub_len)])
        
        return phrases_indices

    def prepare_latents(self, height, width, reward_model=None, generator=None):
        self.height = height
        self.width = width
        self.latent_h, self.latent_w = int(height) // (self.pipe.vae_scale_factor * 2), int(width) // (self.pipe.vae_scale_factor * 2)
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        if self.init_sampling_method is not None:
            batch_size = self.init_sampling_method.cfg.num_chains
        else:
            batch_size = self.cfg.num_particles

        latents, latent_image_ids = self.pipe.prepare_latents(
            batch_size = batch_size,
            num_channels_latents = num_channels_latents,
            height = height,
            width = width,
            dtype = self.pipe.dtype,
            device = self.device,
            generator = generator)
        self.latent_image_ids = latent_image_ids

        if self.init_sampling_method is not None:
            assert reward_model is not None, "grad_log_prob must be provided when using MCMC"
            latents = self.init_sampling_method(latents, reward_model=reward_model, pipe=self)

        return latents
    
    def predict(self, latents, t):
        vel_pred = list()
        for i in range(0, latents.shape[0], self.cfg.mini_batch_size):
            cur_batch_size = min(self.cfg.mini_batch_size, latents.shape[0] - i)
            cur_latents = latents[i:i+cur_batch_size]
            cur_t = t[i:i+cur_batch_size].to(latents.dtype)

            cur_guidance = self.guidance.expand(cur_latents.shape[0]) if self.guidance is not None else None
            cur_pooled_prompt_embeds = self.pooled_peompt_embeds.repeat(cur_batch_size, *([1] * (self.pooled_peompt_embeds.dim() - 1)))
            cur_prompt_embeds = self.prompt_embeds.repeat(cur_batch_size, *([1] * (self.prompt_embeds.dim() - 1)))

            cur_vel_pred = self.pipe.transformer(
                hidden_states = cur_latents,
                timestep = cur_t,
                guidance = cur_guidance,
                pooled_projections = cur_pooled_prompt_embeds,
                encoder_hidden_states = cur_prompt_embeds,
                txt_ids = self.text_ids,
                img_ids = self.latent_image_ids,
                joint_attention_kwargs = {},
                return_dict = False)[0]

            if self.do_true_cfg:
                assert self.negative_prompt_embeds is not None, "Negative prompt embeddings must be encoded first."

                cur_neg_pooled_prompt_embeds = self.negative_pooled_prompt_embeds.repeat(cur_batch_size, *([1] * (self.negative_pooled_prompt_embeds.dim() - 1)))
                cur_neg_prompt_embeds = self.negative_prompt_embeds.repeat(cur_batch_size, *([1] * (self.negative_prompt_embeds.dim() - 1)))

                cur_neg_vel_pred = self.pipe.transformer(
                    hidden_states = cur_latents,
                    timestep = cur_t,
                    guidance = cur_guidance,
                    pooled_projections = cur_neg_pooled_prompt_embeds,
                    encoder_hidden_states = cur_neg_prompt_embeds,
                    txt_ids = self.text_ids,
                    img_ids = self.latent_image_ids,
                    joint_attention_kwargs = {},
                    return_dict = False)[0]
                cur_vel_pred = cur_neg_vel_pred + self.cfg.true_cfg_scale * (cur_vel_pred - cur_neg_vel_pred)
            vel_pred.append(cur_vel_pred)

        vel_pred = torch.cat(vel_pred, dim=0)
        return vel_pred
    
    def forward(self, latents, t):
        assert t.dtype == torch.float32, f"t must be float32, but got {t.dtype}"
        assert latents.shape[0] == t.shape[0], "time must be given in batch manner"
        assert self.prompt_embeds is not None, "Prompt embeddings must be encoded first."

        if not self.do_scheduler_conversion:
            return self.predict(latents, t)
        
        else:
            r = t.clone()
            r_scheduler_output = self.new_scheduler(t=r)

            alpha_r = r_scheduler_output.alpha_t
            sigma_r = r_scheduler_output.sigma_t
            d_alpha_r = r_scheduler_output.d_alpha_t
            d_sigma_r = r_scheduler_output.d_sigma_t
            ###############################################################
            t = self.original_scheduler.snr_inverse(alpha_r / sigma_r)
            t_scheduler_output = self.original_scheduler(t=t)

            alpha_t = t_scheduler_output.alpha_t
            sigma_t = t_scheduler_output.sigma_t
            d_alpha_t = t_scheduler_output.d_alpha_t
            d_sigma_t = t_scheduler_output.d_sigma_t
            ###############################################################
            s_r = sigma_r / sigma_t

            dt_r = (
                sigma_t
                * sigma_t
                * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
                / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
            )

            ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

            s_r = s_r.to(latents.dtype)
            u_t = self.predict(latents / s_r, t)
            u_r = (ds_r * latents / s_r + dt_r * s_r * u_t).to(latents.dtype)
            return u_r
    
    def get_tweedie(self, latents, vel_pred, t):
        assert t.dtype == torch.float32, f"t must be float32, but got {t.dtype}"
    
        cur_scheduler = self.new_scheduler if self.do_scheduler_conversion else self.original_scheduler

        cur_scheduler_output = cur_scheduler(t=t)

        alpha = cur_scheduler_output.alpha_t.to(vel_pred.dtype)
        sigma = cur_scheduler_output.sigma_t.to(vel_pred.dtype)
        d_alpha = cur_scheduler_output.d_alpha_t.to(vel_pred.dtype)
        d_sigma = cur_scheduler_output.d_sigma_t.to(vel_pred.dtype)

        numer = (sigma * vel_pred) - (d_sigma * latents)
        denom = (d_alpha * sigma) - (d_sigma * alpha)

        return numer / denom
    
    def convert_velocity_to_score(self, latents, vel_pred, t):
        cur_scheduler = self.new_scheduler if self.do_scheduler_conversion else self.original_scheduler

        cur_scheduler_output = cur_scheduler(t=t)

        alpha = cur_scheduler_output.alpha_t.to(vel_pred.dtype)
        sigma = cur_scheduler_output.sigma_t.to(vel_pred.dtype)
        d_alpha = cur_scheduler_output.d_alpha_t.to(vel_pred.dtype)
        d_sigma = cur_scheduler_output.d_sigma_t.to(vel_pred.dtype)

        ratio = alpha / d_alpha
        var = sigma ** 2 - ratio * d_sigma * sigma

        ratio = ratio.reshape(ratio.shape[0], *(1,) * (latents.dim() - 1))
        var = var.reshape(var.shape[0], *(1,) * (latents.dim() - 1))

        score = (ratio * vel_pred - latents) / var
        return score
    
    def get_diffuse_coefficient(self, t):
        if self.cfg.sample_method == "ode":
            return torch.zeros_like(t)
        elif self.cfg.sample_method == "sde":
            if self.cfg.diffuse_coeff_func == 'ddpm':
                assert self.cfg.new_scheduler == 'vp', "DDPM diffuse coeff function requires VP conversion"
                cur_alpha_bar = (self.new_scheduler(t=t).alpha_t) ** 2.0
            elif self.cfg.diffuse_coeff_func == "rev_original":
                return self.cfg.diffusion_norm * (1.0 - (t - 1.0)** 2.0)
            else:
                return (1. - self.cfg.diffusion_norm * (t ** 2.0))
        else:
            raise ValueError(f"Unknown sample method {self.cfg.sample_method}. Use 'ode' or 'sde'.")
        
    def get_drift_coefficient(self, latents, vel_pred, t, diffuse_coeff):
        if self.cfg.sample_method == "ode":
            # We assume that dt is always positive
            return -vel_pred
        elif self.cfg.sample_method == "sde":
            score = self.convert_velocity_to_score(latents, vel_pred, t)
            return -vel_pred + (0.5 * diffuse_coeff ** 2) * score
        else:
            raise ValueError(f"Unknown sample method {self.cfg.sample_method}. Use 'ode' or 'sde'.")

    def step(self, latents, t, dt, vel_pred, return_mu_sigma=False):
        assert torch.all(dt >= 0.0).item(), f"dt must be positive"
        assert t.dtype == torch.float32, f"t must be float32, but got {t.dtype}"
        assert dt.dtype == torch.float32, f"dt must be float32, but got {dt.dtype}"

        dt = dt.reshape(dt.shape[0], *(1,) * (latents.dim() - 1))
        diffuse_coeff = self.get_diffuse_coefficient(t)
        diffuse_coeff = diffuse_coeff.reshape(diffuse_coeff.shape[0], *(1,) * (latents.dim() - 1))
        drift_coeff = self.get_drift_coefficient(latents, vel_pred, t, diffuse_coeff)

        next_latents_mean = latents.to(torch.float32) + drift_coeff * dt
        dw = torch.randn_like(latents) * torch.sqrt(dt)
        next_latents = next_latents_mean + diffuse_coeff * dw
        
        if return_mu_sigma:
            return next_latents.to(self.pipe.dtype), next_latents_mean.to(self.pipe.dtype), (diffuse_coeff * torch.sqrt(dt)).to(self.pipe.dtype)
        else:
            return next_latents.to(self.pipe.dtype)
        
    def reverse_step(self, latents, t, dt):
        s = t - dt
        cur_scheduler = self.new_scheduler if self.do_scheduler_conversion else self.original_scheduler

        cur_scheduler_output = cur_scheduler(t=t)
        next_scheduler_output = cur_scheduler(t=s)

        alpha_t = cur_scheduler_output.alpha_t.to(latents.dtype)
        sigma_t = cur_scheduler_output.sigma_t.to(latents.dtype)
        alpha_s = next_scheduler_output.alpha_t.to(latents.dtype)
        sigma_s = next_scheduler_output.sigma_t.to(latents.dtype)
        
        alpha_t_s = alpha_t / alpha_s
        sigma_t_s = (sigma_t**2 - alpha_t_s**2 * sigma_s**2) ** 0.5
        
        latents = latents * alpha_t_s + sigma_t_s * torch.randn_like(latents)
        return latents.to(self.pipe.dtype)

    def decode_latents(self, latents, output_type="pil"):
        # If output_type is "pt", it returns a tensor with dtype torch.bfloat16

        assert self.height is not None and self.width is not None, "Check height and width are initialized"
        assert output_type in ["pt", "pil"], f"output_type must be 'pt' or 'pil', but got {output_type}"
        decoded_images = list()

        for i in range(0, latents.shape[0], self.cfg.mini_batch_size):
            cur_batch_size = min(self.cfg.mini_batch_size, latents.shape[0] - i)
            cur_latents = latents[i:i+cur_batch_size]

            cur_latents = self.pipe._unpack_latents(cur_latents, self.height, self.width, self.pipe.vae_scale_factor)
            cur_latents = (cur_latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            cur_images = self.pipe.vae.decode(cur_latents, return_dict=False)[0]
            cur_images = self.pipe.image_processor.postprocess(cur_images, output_type=output_type)
            decoded_images.extend(cur_images) if output_type == "pil" else decoded_images.append(cur_images)

        if output_type == "pt":
            decoded_images = torch.cat(decoded_images, dim=0)

        return decoded_images

    def encode_images(self, img):
        # If output_type is "pt", it returns a tensor with dtype torch.bfloat16

        assert self.height is not None and self.width is not None, "Check height and width are initialized"
        encoded_latents = list()

        for i in range(0, img.shape[0], self.cfg.mini_batch_size):
            cur_batch_size = min(self.cfg.mini_batch_size, img.shape[0] - i)
            cur_imgs = img[i:i+cur_batch_size]

            image_latents = retrieve_latents(self.pipe.vae.encode(cur_imgs))
            image_latents = (image_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            image_latents = self.pipe._pack_latents(image_latents, cur_batch_size, self.pipe.vae_scale_factor*2, self.cfg.height // self.pipe.vae_scale_factor, self.cfg.width // self.pipe.vae_scale_factor)
            encoded_latents.append(image_latents)

        encoded_latents = torch.cat(encoded_latents, dim=0)

        return encoded_latents

    def decode_latents_no_normalize(self, latents):
        decoded_pt = self.decode_latents(latents, output_type="pt")
        return 2.0 * (decoded_pt) - 1.0
    
    def get_reward_grad_vel_tweedies(self, latents, reward_model, t, reno_reg=False, return_grad=True):
        reward_list = list()
        grad_list = list()
        vel_pred_list = list()
        tweedie_list = list()

        for i in range(0, latents.shape[0], self.cfg.grad_minibatch_size):
            cur_batch_size = min(self.cfg.grad_minibatch_size, latents.shape[0] - i)
            cur_latents = latents[i : i + cur_batch_size].clone().detach().requires_grad_()
            cur_t = t[i : i + cur_batch_size]
            cur_vel_pred = self.forward(cur_latents, cur_t)
            tweedie = self.get_tweedie(cur_latents, cur_vel_pred, cur_t)
            if not reward_model.cfg.decode_to_unnormalized:
                decoded_tweedies = self.decode_latents(tweedie, output_type="pt")
            else:
                decoded_tweedies = self.decode_latents_no_normalize(tweedie)

            cur_reward_values = reward_model(decoded_tweedies.to(torch.float32), self)  # torch.float32

            if return_grad:
                cur_grad = torch.autograd.grad(cur_reward_values, cur_latents, torch.ones_like(cur_reward_values), allow_unused=True)[0]
                cur_grad = cur_grad.nan_to_num()
                if reward_model.cfg.grad_norm is not None:
                    cur_grad = cur_grad.to(torch.float32)
                    grad_scale = torch.mean(cur_grad ** 2) ** 0.5
                    if grad_scale > reward_model.cfg.grad_norm:
                        cur_grad = cur_grad * (reward_model.cfg.grad_norm / grad_scale)
                    cur_grad = cur_grad.to(latents.dtype)
                elif reward_model.cfg.grad_const_scale is not None:
                    cur_grad = cur_grad * reward_model.cfg.grad_const_scale

            reward_list.append(cur_reward_values.detach().clone())
            grad_list.append(cur_grad.detach().clone()) if return_grad else None
            vel_pred_list.append(cur_vel_pred.detach().clone())
            tweedie_list.append(decoded_tweedies.detach().clone())

            del cur_vel_pred, tweedie, decoded_tweedies, cur_reward_values
            
        rewards = torch.cat(reward_list, dim=0)
        grads = torch.cat(grad_list, dim=0) if return_grad else None
        vel_preds = torch.cat(vel_pred_list, dim=0)
        tweedies = torch.cat(tweedie_list, dim=0)
        if reward_model.cfg.decode_to_unnormalized:
            tweedies = tweedies * 0.5 + 0.5
        return rewards, grads, vel_preds, tweedies

    def get_reward_grad(self, latents, reward_model, is_pixel_space=False, return_grad=True):
        reward_list = list()
        grad_list = list()

        for i in range(0, latents.shape[0], self.cfg.grad_minibatch_size):
            cur_batch_size = min(self.cfg.grad_minibatch_size, latents.shape[0] - i)
            cur_latents = latents[i : i + cur_batch_size].detach().requires_grad_()
            
            if not is_pixel_space:
                if not reward_model.cfg.decode_to_unnormalized:
                    decoded_tweedies = self.decode_latents(cur_latents, output_type="pt")
                else:
                    decoded_tweedies = self.decode_latents_no_normalize(cur_latents)
            else:
                decoded_tweedies = cur_latents

            cur_reward_values = reward_model(decoded_tweedies.to(torch.float32), self)  # torch.float32

            if return_grad:
                cur_grad = torch.autograd.grad(cur_reward_values, cur_latents, torch.ones_like(cur_reward_values), allow_unused=True)[0]
                cur_grad = cur_grad.nan_to_num()
                if reward_model.cfg.grad_norm is not None:
                    cur_grad = cur_grad.to(torch.float32)
                    grad_scale = torch.mean(cur_grad ** 2) ** 0.5
                    if grad_scale > reward_model.cfg.grad_norm:
                        cur_grad = cur_grad * (reward_model.cfg.grad_norm / grad_scale)
                    cur_grad = cur_grad.to(latents.dtype)
                elif reward_model.cfg.grad_const_scale is not None:
                    cur_grad = cur_grad * reward_model.cfg.grad_const_scale
                
            reward_list.append(cur_reward_values.detach().clone())
            grad_list.append(cur_grad.detach().clone()) if return_grad else None

            del decoded_tweedies, cur_reward_values

        rewards = torch.cat(reward_list, dim=0)
        grads = torch.cat(grad_list, dim=0) if return_grad else None
        return rewards, grads


    def set_custom_call_function_for_MCMC(self, custom_call_function):
        assert self.init_sampling_method is not None, "Custom call function can only be set when using MCMC."
        self.init_sampling_method.custom_call_function = custom_call_function

    def set_mcmc_save_dir(self, save_dir):
        assert self.init_sampling_method is not None, "MCMC save dir can only be set when using MCMC."
        self.init_sampling_method.cfg.save_mcmc_dir = save_dir