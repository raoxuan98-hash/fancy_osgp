import copy
import torch
from torch import nn
from copy import deepcopy
import timm
from models.sgp_lora import SGPLoRACLIPVisionTransformer
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_clip_model(args, train_mode="lora"):
    """
    train_mode: "lora" | "full" | "frozen"
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    if train_mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        return model, processor

    elif train_mode == "full":
        for n, p in model.named_parameters():
            if "vision_model.encoder.layers" in n and ("self_attn" in n or "mlp" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
        return model, processor

    elif train_mode == "lora":
        for p in model.parameters():
            p.requires_grad = False

        rank = args['lora_rank']

        if args['lora_type'] == 'nsp_lora':
            use_soft_projection = False
        
        elif args['lora_type'] == "sgp_lora":
            use_soft_projection = True

        model.vision_model = SGPLoRACLIPVisionTransformer(
            model.vision_model,
            r=rank,
            weight_temp=args['weight_temp'],
            use_soft_projection=use_soft_projection,
            weight_kind=args['weight_kind'],
            weight_p=args['weight_p'],
            nsp_eps=args['nsp_eps'],
            nsp_weight=args['nsp_weight'])
        
        return model, processor

    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")


class CLIP_BaseNet(nn.Module):
    def __init__(self, args, train_mode="lora"):
        super(CLIP_BaseNet, self).__init__()
        self.train_mode = train_mode

        self.model, self.processor = get_clip_model(args, train_mode=train_mode)

        self.valid_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])])

    def forward(self, img, text):
        x = self.model.get_image_features(img)
        y = self.model.get_text_features(text)
        return x, y

    def encode_image(self, img):
        return self.model.get_image_features(img)

    def encode_text(self, text):
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.model.device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        return text_features

    @property
    def feature_dim(self):
        return self.model.config.projection_dim  # CLIP 输出维度，通常是 512