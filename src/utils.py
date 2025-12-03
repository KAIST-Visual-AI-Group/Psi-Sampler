import os
import sys
from dataclasses import fields
import random
import math
from numba import jit
import gc

from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from torch.nn.functional import interpolate
from torchvision.utils import make_grid

from omegaconf import OmegaConf, DictConfig
from PIL import Image, ImageDraw


class StrictStdoutSuppressor:
    def __init__(self, allowed_prefix="[MYPRINT]"):
        self.allowed_prefix = allowed_prefix
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def write(self, msg):
        if msg.startswith(self.allowed_prefix):
            self.original_stdout.write(msg[len(self.allowed_prefix):])
        return

    def flush(self):
        self.original_stdout.flush()
        self.original_stderr.flush()

def print_my(*args, **kwargs):
    msg = ' '.join(str(arg) for arg in args)
    sys.__stdout__.write(msg + '\n')  
    sys.stdout.write("[MYPRINT]" + msg + '\n')  

def suppress_print():
    return StrictStdoutSuppressor()

def ignore_kwargs(cls):
    original_init = cls.__init__

    def init(self, *args, **kwargs):
        expected_fields = {field.name for field in fields(cls)}
        expected_kwargs = {
            key: value for key, value in kwargs.items() if key in expected_fields
        }
        original_init(self, *args, **expected_kwargs)

    cls.__init__ = init
    return cls

def load_config(*yamls, cli_args = None, from_string=False, **kwargs):
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    return cfg

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor2PIL(tensor, do_normalize=False):
    tensor = tensor.clone().detach().cpu().float()
    if do_normalize:
        tensor = tensor / 2.0 + 0.5
    tensor = tensor.clip(0, 1).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    tensor = np.transpose(tensor, (1, 2, 0))
    image = Image.fromarray(tensor)
    return image

def save_collage(tensors, save_path, rows=None, cols=None, resize=(256, 256), reward_values=None, bboxes=None, phrases=None):
    tensors = tensors.cpu().float()
    B, C, H, W = tensors.shape
    resized = interpolate(tensors, size=resize, mode='bilinear', align_corners=False)

    if reward_values is not None:
        assert len(reward_values) == B, f"scores should have length {B}, but got {len(reward_values)}"
        if isinstance(reward_values, torch.Tensor):
            reward_values = reward_values.cpu().float()

    images_with_text = []
    to_pil = T.ToPILImage()
    for i in range(B):
        img = to_pil(resized[i])
        if reward_values is not None:
            draw = ImageDraw.Draw(img)
            text = (
                f"{reward_values[i].item():.5f}" if isinstance(reward_values[i], torch.Tensor)
                else f"{reward_values[i]:.5f}" if isinstance(reward_values[i], float)
                else f"{reward_values[i]}" if isinstance(reward_values[i], int)
                else reward_values[i]
            )
            # text = f"{reward_values[i]:.5f}"
            draw.rectangle([0, 0, 120, 20], fill=(0, 0, 0, 128))  
            draw.text((5, 2), text, fill=(255, 255, 255))
        
        if bboxes is not None and phrases is not None:
            h, w = resize
            scaled_bboxes = [[x1 * w / W, y1 * h / H, x2 * w / W, y2 * h / H] for x1, y1, x2, y2 in bboxes]
            draw_box(img, scaled_bboxes, phrases, h, w)  
            
        images_with_text.append(T.ToTensor()(img)[None, ...])

    if rows is None or cols is None:
        n = math.ceil(math.sqrt(B))
        total = n * n
        leftover = total - B
    else:
        n = cols
        total = rows * cols
        leftover = total - B

    if leftover > 0:
        pad = torch.zeros((leftover, C, resize[0], resize[1]))
        images_with_text += [pad]

    stacked = torch.cat(images_with_text[:total], dim=0)
    grid_tensor = make_grid(stacked, nrow=n, padding=2)
    pil_image = T.ToPILImage()(grid_tensor)
    pil_image.save(save_path)
    
# Originated from https://github.com/KAIST-Visual-AI-Group/GrounDiT/blob/main/groundit/utils.py
def draw_box(pil_img, bboxes, phrases, height, width):
    """
    Draws bounding boxes with associated phrases on a PIL image.

    Args:
        pil_img (PIL.Image.Image): The image to draw on.
        bboxes (list of list): Bounding boxes, where each box is represented by [ul_x, ul_y, lr_x, lr_y].
            For the bbox convention, see the `sanity_check` function in this file.
        phrases (list of str): Semicolon-separated phrases corresponding to the bounding boxes.
        height (int): Height of the image.
        width (int): Width of the image.
    """
    draw = ImageDraw.Draw(pil_img)
    
    # Iterate over bounding boxes and phrases
    for obj_bbox, phrase in zip(bboxes, phrases):
        # Assume bboxes in 512 x 512 pixel space
        height_ratio = height / 512
        width_ratio = width / 512
        obj_bbox = [int(coord * height_ratio) if i % 2 == 0 else int(coord * width_ratio) for i, coord in enumerate(obj_bbox)]
        # Draw the bounding box
        draw.rectangle(obj_bbox, outline="red", width=5)
        
        # Draw the associated phrase
        draw.text((obj_bbox[0] + 5, obj_bbox[1] + 5), phrase, fill=(255, 0, 0))

### SMC utils ###

@jit(nopython=True)
def ssp(W, M):
    """SSP resampling.

    SSP stands for Srinivasan Sampling Process. This resampling scheme is
    discussed in Gerber et al (2019). Basically, it has similar properties as
    systematic resampling (number of off-springs is either k or k + 1, with
    k <= N W^n < k +1), and in addition is consistent. See that paper for more
    details.

    Reference
    =========
    Gerber M., Chopin N. and Whiteley N. (2019). Negative association, ordering
    and convergence of resampling methods. Ann. Statist. 47 (2019), no. 4, 2236–2260.
    """
    N = W.shape[0]
    MW = M * W
    nr_children = np.floor(MW).astype(np.int64)
    xi = MW - nr_children
    u = np.random.rand(N - 1)
    i, j = 0, 1
    for k in range(N - 1):
        delta_i = min(xi[j], 1.0 - xi[i])  # increase i, decr j
        delta_j = min(xi[i], 1.0 - xi[j])  # the opposite
        sum_delta = delta_i + delta_j
        # prob we increase xi[i], decrease xi[j]
        pj = delta_i / sum_delta if sum_delta > 0.0 else 0.0
        # sum_delta = 0. => xi[i] = xi[j] = 0.
        if u[k] < pj:  # swap i, j, so that we always inc i
            j, i = i, j
            delta_i = delta_j
        if xi[j] < 1.0 - xi[i]:
            xi[i] += delta_i
            j = k + 2
        else:
            xi[j] -= delta_i
            nr_children[i] += 1
            i = k + 2
    # due to round-off error accumulation, we may be missing one particle
    if np.sum(nr_children) == M - 1:
        last_ij = i if j == k + 2 else j
        if xi[last_ij] > 0.99:
            nr_children[last_ij] += 1
    if np.sum(nr_children) != M:
        # file a bug report with the vector of weights that causes this
        raise ValueError("ssp resampling: wrong size for output")
    return np.arange(N).repeat(nr_children)


def cumsum_deterministic_1d(tensor):
    """Deterministic cumsum (1D) using explicit loop (avoids CUDA non-determinism)."""
    result = torch.zeros_like(tensor)
    acc = torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    for i in range(tensor.shape[0]):
        acc += tensor[i]
        result[i] = acc
    return result

def deterministic_multinomial(normalized_weights, num_particles, generator):
    cdf = cumsum_deterministic_1d(normalized_weights)

    u = torch.rand(num_particles, generator=generator, device=normalized_weights.device)

    # inverse-CDF
    ancestor_idx = torch.searchsorted(cdf, u)
    return ancestor_idx

def filter_mismatched_weights(model_state_dict, weight_state_dict):
    mismatch_keys = {}
    for key in list(model_state_dict.keys()):
        if key in weight_state_dict:
            value_model = model_state_dict[key]
            value_state_dict = weight_state_dict[key]
            if value_model.shape != value_state_dict.shape:
                weight_state_dict[key] = value_model
                mismatch_keys[key] = [value_model.shape, value_state_dict.shape]
    return weight_state_dict, mismatch_keys

def load_state_dict(model, state_dict):
    if state_dict is None:
        return

    # load _classes_ for inference
    if "_classes_" in state_dict:
        dummy_classes = torch.zeros_like(state_dict["_classes_"])
        model.register_buffer("_classes_", dummy_classes)

    # initialize keys list
    matched_state_dict, mismatch_keys = filter_mismatched_weights(model.state_dict(), state_dict)
    incompatible_keys = model.load_state_dict(matched_state_dict, strict=False)

def cast_model_to_bf16(model: torch.nn.Module):
    """
    Recursively cast all floating‐point parameters and buffers of `model` to bfloat16.
    """
    # 1) cast parameters
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point:
            param.data = param.data.to(torch.bfloat16)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(torch.bfloat16)
    # 2) cast buffers (e.g. LayerNorm running stats)
    for name, buf in model.named_buffers():
        if buf.dtype.is_floating_point:
            model.register_buffer(name, buf.to(torch.bfloat16))



def eval_model_gpu_reward_cpu(eval_models, reward_model, device):
    for i in range(len(eval_models)):
        if eval_models[i] is not None:
            eval_models[i] = eval_models[i].to(device)
            
    reward_model = reward_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return eval_models, reward_model
    
def eval_model_cpu_reward_gpu(eval_models, reward_model, device):
    for i in range(len(eval_models)):
        if eval_models[i] is not None:
            eval_models[i] = eval_models[i].to("cpu")
    
    torch.cuda.empty_cache()
    gc.collect()
    reward_model = reward_model.to(device)
    return eval_models, reward_model


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")