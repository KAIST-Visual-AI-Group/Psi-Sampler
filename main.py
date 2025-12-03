import os
import argparse
from dataclasses import dataclass
from PIL import ImageDraw
from tqdm import tqdm
import json

from src.utils import *
from src.flux_pipeline import StochasticFluxPipeline, TimeSampler
from src.runner import *
from src.reward_model import *
from src.mcmc import pCNL


@ignore_kwargs
@dataclass
class Config:
    seed: int = 0
    negative_prompt: str = None
    height: int = 512
    width: int = 512

def main(main_cfg, CFG, args, task_name):
    device = torch.device("cuda:0")

    with suppress_print():
        time_sampler = TimeSampler(device, CFG)
        pipe = StochasticFluxPipeline(device, CFG)
        reward_model = get_reward_model(task_name)(torch.float32, device, args.save_dir, CFG)
        SMC_runner = SMC(CFG)

    if args.data_path.endswith(".json"):
        dataset = json.load(open(args.data_path, 'r'))
    else:
        with open(args.data_path, 'r') as f:
            dataset = [line.strip() for line in f if line.strip() != '']
    
    for idx, data in enumerate(tqdm(dataset, total=len(dataset), desc="Benchmark")):
        data = data if isinstance(dataset, list) else dataset[data]
        prompt = data if isinstance(data, str) else data["prompt"]
        phrases = data['phrases'] if isinstance(data, dict) and "phrases" in data else None

        pipe.load_encoder()
        pipe.encode_prompt(prompt, main_cfg.negative_prompt, phrases=phrases)
        reward_model.register_data(data)
        pipe.unload_encoder()

        seed_everything(main_cfg.seed)
        generator = torch.Generator(device=device).manual_seed(main_cfg.seed)
        
        reward_model.cfg.grad_const_scale = CFG.grad_const_scale
        reward_model.cfg.grad_norm = CFG.grad_norm
        
        # pCNL
        latents = pipe.prepare_latents(height=main_cfg.height, width=main_cfg.width, reward_model=reward_model, generator=generator)

        reward_model.cfg.grad_norm = CFG.smc_grad_norm
        reward_model.cfg.grad_const_scale = CFG.smc_grad_const_scale

        # SMC
        sample, sample_reward = SMC_runner.run(pipe, time_sampler, reward_model, latents, generator, idx=idx)
        final_latent = pipe.decode_latents(sample.detach(), output_type="pt")
        
        image = torchvision.transforms.ToPILImage()(final_latent[0].float().cpu().clamp(0, 1))
        image.save(os.path.join(args.save_dir, f"{idx:05d}.png"))
        if args.save_reward:
            draw = ImageDraw.Draw(image)
            text = f"{sample_reward.item():.5f}" if hasattr(sample_reward, "item") else f"{sample_reward:.5f}"
            draw.rectangle([0, 0, 120, 20], fill=(0, 0, 0, 128))  
            draw.text((5, 2), text, fill=(255, 255, 255))
            
            if "layout_to_image" in task_name:
                draw_box(image, data['bboxes'], phrases, main_cfg.height, main_cfg.width)
                
            image.save(os.path.join(args.save_dir, "img_rewards", f"{idx:05d}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/layout_to_image.yaml")
    parser.add_argument("--data_path", default="./data/layout_to_image.json")
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument("--save_tweedies", action="store_true", help="Save the tweedies")
    parser.add_argument("--save_reward", action="store_true", help="Save the reward value on the image")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--extra_tag", default=None)


    args, extras = parser.parse_known_args()
    CFG = load_config(args.config, cli_args=extras)

    task_name = args.config.split("/")[-1].split(".")[0]
    main_cfg = Config(**CFG)

    step_size = CFG.step_size
    rho = pCNL.get_rho(step_size)
    CFG.rho = rho
    grad_norm = CFG.grad_norm
    smc_grad_norm = CFG.smc_grad_norm
    mcmc_name = CFG.mcmc
        
    name = f"{mcmc_name}_test"
    
    if args.extra_tag is not None:
        name = name + "_" + args.extra_tag

    qualitative_dir = os.path.join(args.save_dir, task_name) if args.tag is None else os.path.join(args.save_dir, task_name, args.tag)
    args.save_dir = os.path.join(qualitative_dir, name)
    args.misc_dir = os.path.join(args.save_dir, "misc")
    os.makedirs(args.misc_dir, exist_ok=True)

    with open(os.path.join(args.misc_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(CFG))

    CFG.save_dir = args.save_dir
    CFG.qualitative_dir = qualitative_dir
    CFG.misc_dir = args.misc_dir
    CFG.name = name
    CFG.save_tweedies = args.save_tweedies
    CFG.reward_name = task_name
    if args.save_reward:
        os.makedirs(os.path.join(args.save_dir, "img_rewards"), exist_ok=True)
    main(main_cfg, CFG, args, task_name)


# Example Commands

# 1. Layout-to-Image
# python main.py --tag layout_to_image --config ./config/layout_to_image.yaml --data_path ./data/layout_to_image.json --save_dir ./results_layout_to_image --save_reward --save_tweedies

# 2. Aesthetic Preference
# python main.py --tag aesthetic --config ./config/aesthetic.yaml --data_path ./data/aesthetic.txt --save_dir ./results_aesthetic --save_reward --save_tweedies

# 3. Quantity-Aware Image Generation
# TODO