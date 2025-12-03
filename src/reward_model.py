import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import os
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from transformers import CLIPTokenizer
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPModel, CLIPProcessor

from dataclasses import dataclass
from src.utils import ignore_kwargs

from src.t2icount.models.reg_model import Count


__rewards__ = dict()

def register_reward_model(name):
    def decorator(cls):
        __rewards__[name] = cls
        return cls
    return decorator

def get_reward_model(name: str):
    if name not in __rewards__:
        raise ValueError(f"Reward model {name} not found. Available reward models: {list(__rewards__.keys())}")
    return __rewards__[name]

@register_reward_model(name="aesthetic")
class AestheticRewardModel(nn.Module):
    @ignore_kwargs
    @dataclass
    class Config():
        decode_to_unnormalized: bool = False 
        grad_norm: float = None
        grad_const_scale: float = None

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )

        def forward(self, embed):
            return self.layers(embed)

    def __init__(self, dtype, device, save_dir, CFG):
        super().__init__()
        self.cfg = self.Config(**CFG)
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device, dtype=self.dtype)
        self.mlp = self.MLP().to(self.device, dtype=self.dtype)
        state_dict = torch.load("./misc/aesthetic_score/sac+logos+ava1-l14-linearMSE.pth", map_location=self.device)
        self.mlp.load_state_dict(state_dict)
        self.target_size =  224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                          std=[0.26862954, 0.26130258, 0.27577711])
        self.toPIL = T.ToPILImage()
        self.eval()
        self.requires_grad_(False)

    def __call__(self, images, _):
        inputs = F.interpolate(images, (self.target_size, self.target_size), mode='bilinear', align_corners=False)
        inputs = self.normalize(inputs).to(self.dtype)
        embed = self.clip.get_image_features(pixel_values=inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        score = self.mlp(embed).squeeze(1)
        return score
    
    def register_data(self, data):
        pass

@register_reward_model(name="layout_to_image")
class LayoutToImageReward(nn.Module):
    @ignore_kwargs
    @dataclass
    class Config:
        decode_to_unnormalized: bool = True
        grad_norm: float = None
        grad_const_scale: float = None
        
        height: int = 512,
        width: int = 512,
        
        save_vram : bool = True

    def __init__(self, dtype, device, save_dir, CFG):
        super().__init__()
        self.cfg = self.Config(**CFG)
        self.device = device
        self.dtype = dtype
        self.save_dir = save_dir

        self.gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.object_detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(self.device)
        
        self.requires_grad_(False)
        self.eval()

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.phrases = None
        self.target_bboxes = None  # normalized [x1, y1, x2, y2]

    def register_data(self, data):
        self.phrases = data["phrases"]
        self.target_bboxes = torch.tensor(data["bboxes"], device=self.device, dtype=torch.float32)
        self.relations = data['relations'] if 'relations' in data else None
        if self.target_bboxes.max() > 1:
            # assume input is in 512 x 512 pixels, convert to normalized
            self.target_bboxes[:, [0, 2]] /= 512
            self.target_bboxes[:, [1, 3]] /= 512
    
    def phrases_to_indices(self):
        phrase_to_indices = defaultdict(list)
        for i, phrase in enumerate(self.phrases):
            phrase_to_indices[phrase].append(i)
        return phrase_to_indices

    def forward(self, x, pipe=None):
        assert x.shape[0] == 1, "Only batch size 1 supported"
        x = x.to(self.device).to(self.dtype)
        processed_image = self.normalize(x)

        text = ". ".join(self.phrases) + "."
        gd_inputs = self.gd_processor.tokenizer(
            text,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.object_detector(
            pixel_values=processed_image,
            input_ids=gd_inputs["input_ids"],
            attention_mask=gd_inputs["attention_mask"]
        )

        logits = outputs.logits       # (1, num_queries, vocab_size)
        pred_boxes = outputs.pred_boxes  # (1, num_queries, 4)

        phrase_to_indices = self.phrases_to_indices()
        input_ids = gd_inputs["input_ids"][0]
        phrase_token_map = self.get_phrase_token_indices(input_ids, list(phrase_to_indices.keys()))

        matched_boxes = []
        used_pred_indices = set()
        gt_bboxes = self.target_bboxes.to(self.device).to(self.dtype)
        top_k = 10
        score_thresh = 0.1

        for i, (phrase, gt_indices) in enumerate(phrase_to_indices.items()):
            phrase_idx = phrase_token_map[i]  # get token index for this phrase
            phrase_scores = logits[0, :, phrase_idx]  # (num_queries,)

            sorted_indices = torch.argsort(phrase_scores, descending=True)
            topk_indices = sorted_indices[:top_k]
            valid_mask = phrase_scores[topk_indices] > score_thresh
            valid_topk_indices = topk_indices[valid_mask]

            # fallback to highest score regardless of threshold
            fallback_idx = topk_indices[0].item()
            fallback_box = pred_boxes[0, fallback_idx]

            for gt_idx in gt_indices:
                best_match = None
                best_iou = 0.0
                best_idx = None
                gt_box = gt_bboxes[gt_idx].unsqueeze(0)

                for pred_idx in valid_topk_indices:
                    pred_idx = pred_idx.item()
                    if pred_idx in used_pred_indices:
                        continue
                    pred_box = pred_boxes[0, pred_idx].unsqueeze(0)
                    iou = self.compute_iou(self.cxcywh_to_xyxy(pred_box), gt_box).item()
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pred_box
                        best_idx = pred_idx

                if best_match is not None and best_iou > 0.1:
                    matched_boxes.append(best_match.squeeze(0))
                    used_pred_indices.add(best_idx)
                else:
                    matched_boxes.append(fallback_box)
                    used_pred_indices.add(fallback_idx)

        pred_bboxes = torch.stack(matched_boxes, dim=0)
        final_gt_bboxes = gt_bboxes

        ious = self.compute_iou(self.cxcywh_to_xyxy(pred_bboxes), final_gt_bboxes)
        reward = ious.mean()
        return reward.unsqueeze(0)

    def get_phrase_token_indices(self, input_ids, phrases):
        """
        Match each phrase to a token index in the input_ids.
        Returns one index per phrase (first matching token).
        """
        tokenizer = self.gd_processor.tokenizer 
        phrase_inds = []
        for phrase in phrases:
            phrase_tokens = tokenizer(phrase)["input_ids"][1:-1]  # remove CLS/SEP
            for t in phrase_tokens:
                idx = (input_ids == t).nonzero(as_tuple=True)
                if len(idx[0]) > 0:
                    phrase_inds.append(idx[0][0].item())
                    break
            else:
                phrase_inds.append(0)  
        return phrase_inds

    def cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def compute_giou(self, boxes1, boxes2):
        """
        boxes1, boxes2: (N, 4) in [x1, y1, x2, y2] format (normalized or pixel)
        Returns: GIoU score for each pair (N,)
        """
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter

        iou = inter / (union + 1e-6)

        # compute enclosing box
        x1_c = torch.min(boxes1[:, 0], boxes2[:, 0])
        y1_c = torch.min(boxes1[:, 1], boxes2[:, 1])
        x2_c = torch.max(boxes1[:, 2], boxes2[:, 2])
        y2_c = torch.max(boxes1[:, 3], boxes2[:, 3])
        enclosing = (x2_c - x1_c) * (y2_c - y1_c) + 1e-6

        giou = iou - (enclosing - union) / enclosing
        return giou

    def compute_iou(self, boxes1, boxes2):
        """
        boxes1, boxes2: (N, 4), [x1, y1, x2, y2] normalized
        """
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter
        return inter / (union + 1e-6)


@register_reward_model(name="t2icount")
class T2ICountReward(nn.Module):
    @ignore_kwargs
    @dataclass
    class Config:
        decode_to_unnormalized: bool = True
        grad_norm: float = None
        grad_const_scale: float = None
        
        height: int = 512,
        width: int = 512,
        
        save_vram : bool = True

    def __init__(self, dtype, device, save_dir, CFG):
        super().__init__()
        self.cfg = self.Config(**CFG)
        self.device = device
        self.dtype = dtype
        self.save_dir = save_dir

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_path1 = os.path.join(base_dir, "misc", "t2icount", "v1-5-pruned-emaonly.ckpt")
        ckpt_path2 = os.path.join(base_dir, "misc", "t2icount", "best_model_paper.pth")

        unet_cfg = {'base_size': 384, 'max_attn_size': 384 // 8, 'attn_selector': 'down_cross+up_cross'}
        model = Count('src/t2icount/v1-inference.yaml', ckpt_path1, unet_config=unet_cfg)
        model.load_state_dict(torch.load(ckpt_path2, map_location='cpu'), strict=False)
        self.model = model.to(device).eval()

        self.to_tensor = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def register_data(self, data):
        self.data = data
    
    def make_prompt_mask(self, prompt: str) -> torch.Tensor:
        mask = torch.zeros(77)
        toks = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        L = toks['input_ids'].shape[1]
        mask[1 : 1 + L] = 1 
        return mask                # (77,)
    
    def get_all(self, x, pipe=None):
        cnt_err_sum = 0
        dentisy_maps = []
        pred_counts = []

        x = self.to_tensor(x)
        for datum in self.data["items"]:
            target = torch.tensor([datum["count"]], device=self.device, dtype=torch.float32)

            attn_mask = self.make_prompt_mask(datum["prompt"]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.device)

            output = self.model(
                x,
                [datum["prompt"]],
                attn_mask
            )[0]

            pred = torch.sum(output) / 60.0

            # Aggregate predicted counts per batch.
            pred_counts.append(pred.item())
            
            # smooth l1 loss
            orig = F.smooth_l1_loss(pred, target, beta=1.0, reduction='none')   # (B,)
            cnt_err = orig


            cnt_err_sum = cnt_err_sum + cnt_err
            dentisy_maps.append(output)
        
        dentisy_maps = torch.stack(dentisy_maps, dim=0).squeeze(1)  # (B, 1, 1, H, W) -> (B, 1, H, W)
        max_maps, max_indices = dentisy_maps.max(dim=0)


        # Use the maximum density map for overlay
        combined_overlay = max_maps.unsqueeze(0).clamp(0, 1)
        combined_overlay = F.interpolate(combined_overlay, size=(x.shape[2], x.shape[3]), mode='nearest').squeeze(1)
        return {
            "reward": -cnt_err_sum,
            "overlay": combined_overlay,
            "counts": pred_counts
        }

    def forward(self, x, pipe=None):
        return self.get_all(x, pipe)["reward"]

    def get_overlay(self, x, pipe=None):
        return self.get_all(x, pipe)["overlay"]

    def get_counts(self, x, pipe=None):
        return self.get_all(x, pipe)["counts"]