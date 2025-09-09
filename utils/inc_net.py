import copy
import torch
from torch import nn
from copy import deepcopy
import timm
from lora import NullSpaceViT
from proj_lora import LoRAViT
from models.basic_lora import PlainLoRAViT
from models.osgp_lora import OSGPLoRAViT, SGPLoRAViT
from models.osgp_lora_clip import OSGPLoRAViT_CLIP, SGPLoRAViT_CLIP
# from models.sgp_lora import SGPLoRAViT

rank = 2
subspace_dim = 12
def get_vit(args, pretrained=False):
    name = args['vit_type']
    name = name.lower()
    rank = args['lora_rank']

    if name == 'vit-b-p16':
        vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)

    elif name == 'vit-b-p16-mocov3':
        vit = timm.create_model('vit_base_patch16_224.', pretrained=False, num_classes=0)
        model_dict = torch.load('mocov3-vit-base-300ep.pth', weights_only=False)
        vit.load_state_dict(model_dict['model'], strict=True)
    
    elif name == 'vit-b-p16-dino':
        vit = timm.create_model('vit_base_patch16_224.dino', pretrained=pretrained, num_classes=0)

    elif name == 'vit-b-p16-mae':
        vit = timm.create_model('vit_base_patch16_224.mae', pretrained=pretrained, num_classes=0)

    else:
        raise ValueError(f'Model {name} not supported')
    
    vit.head = nn.Identity()
    
    del vit.norm
    vit.norm = nn.LayerNorm(768, elementwise_affine=False)
    
    lora_type = args['lora_type']
    if lora_type == "full":
        return NullSpaceViT(vit, use_projection=args['use_projection'])
    
    elif lora_type == "basic_lora":
        return PlainLoRAViT(vit, r=rank)

    elif lora_type == "osgp_lora":
        return OSGPLoRAViT(vit, r=rank, proj_temp=args['proj_temp'], kl_gamma=args['kl_gamma'], trace_k=args['trace_k'], weight_p=args['weight_p'])
    
    elif lora_type == "sgp_lora":
        return SGPLoRAViT(vit, r=rank,proj_temp=args['proj_temp'], use_soft_projection=True, k=args['trace_k'], weight_kind=args['weight_kind'], weight_p=args['weight_p'])
    
    elif lora_type == "nsp_lora":
        return SGPLoRAViT(vit, r=rank,proj_temp=args['proj_temp'], use_soft_projection=False, k=args['trace_k'], nsp_eps=args['nsp_eps'], nsp_weight=args['nsp_weight'])


def get_clip_model(args):
    rank = args['lora_rank']

    import clip 
    model, train_preprocess, val_preprocess = clip.load("ViT-B/16", jit=False)

    for parameter in model.visual.parameters():
        parameter.requires_grad = False
    
    for parameter in model.transformer.parameters():
        parameter.requires_grad = False
    
    lora_type = args['lora_type']

    if lora_type == "full":
        model = NullSpaceViT(model.visual, use_projection=args['use_projection'])

    elif lora_type == "osgp_lora":
        model = OSGPLoRAViT_CLIP(model, r=rank, proj_temp=args['proj_temp'], kl_gamma=args['kl_gamma'], trace_k=args['trace_k'], weight_p=args['weight_p'])
    
    elif lora_type == "sgp_lora":
        model = SGPLoRAViT_CLIP(model, r=rank, proj_temp=args['proj_temp'], use_soft_projection=True, k=args['trace_k'], weight_kind=args['weight_kind'], weight_p=args['weight_p'])
    
    elif lora_type == "nsp_lora":
        model = SGPLoRAViT_CLIP(model, r=rank, proj_temp=args['proj_temp'], use_soft_projection=False, k=args['trace_k'], nsp_eps=args['nsp_eps'], nsp_weight=args['nsp_weight'])

    return model, train_preprocess, val_preprocess

class ContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes, bias=False)])
        self.head_weights = nn.Parameter(torch.ones(nb_classes))
        self.current_output_size = nb_classes

    def update(self, nb_classes):
        new_head = nn.Linear(self.embed_dim, nb_classes, bias=False)
        
        self.heads.append(new_head)
        
        new_head_weights = nn.Parameter(torch.ones(self.current_output_size + nb_classes))
        with torch.no_grad():
            new_head_weights[:self.current_output_size] = self.head_weights
            new_head_weights[self.current_output_size:] = 1.0
        
        self.head_weights = new_head_weights
        self.current_output_size += nb_classes

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        combined = torch.cat(outputs, dim=1)
        return combined * self.head_weights


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.vit = get_vit(args, pretrained)
        self.fc = None

    def extract_vector(self, x):
        return self.vit(x)

    def forward(self, x):
        x = self.vit(x)
        out = self.fc(x)
        return out
    
    @property
    def feature_dim(self):
        return self.vit.feature_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = ContinualLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes)

    def copy(self):
        return copy.deepcopy(self)
    

class CLIP_BaseNet(nn.Module):
    def __init__(self, args):
        super(CLIP_BaseNet, self).__init__()
        self.model, self.train_preprocess, self.valid_preprocess = get_clip_model(args)
        self.fc = None

    def forward(self, img, text):
        x = self.model.clip.encode_image(img)
        y = self.model.clip.encode_text(text)
        return x, y
    
    def encode_image(self, img):
        return self.model.clip.encode_image(img)
    
    def encode_text(self, text):
        return self.model.clip.encode_text(text)
    
    @property
    def feature_dim(self):
        return self.model.embed_dim
