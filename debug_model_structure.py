#!/usr/bin/env python3
"""调试CLIP模型结构的脚本"""

import sys
from utils.inc_net import CLIP_BaseNet

def debug_model_structure():
    """调试CLIP模型结构"""
    
    # 创建测试参数
    args = {
        'lora_rank': 4,
        'lora_type': 'sgp_lora',
        'weight_temp': 1.0,
        'weight_kind': 'log1p',
        'weight_p': 1.0,
        'nsp_eps': 0.05,
        'nsp_weight': 0.0
    }
    
    # 创建CLIP模型
    print("创建CLIP模型...")
    model = CLIP_BaseNet(args, train_mode="lora")
    
    print(f"模型类型: {type(model)}")
    print(f"是否有model属性: {hasattr(model, 'model')}")
    
    if hasattr(model, 'model'):
        inner_model = model.model
        print(f"内部模型类型: {type(inner_model)}")
        print(f"内部模型是否有vision_model属性: {hasattr(inner_model, 'vision_model')}")
        
        if hasattr(inner_model, 'vision_model'):
            vision_model = inner_model.vision_model
            print(f"vision_model类型: {type(vision_model)}")
            print(f"vision_model是否有encoder属性: {hasattr(vision_model, 'encoder')}")
            print(f"vision_model是否有clip_vision_model属性: {hasattr(vision_model, 'clip_vision_model')}")
            
            # 检查SGPLoRACLIPVisionTransformer的结构
            if hasattr(vision_model, 'clip_vision_model'):
                clip_vision_model = vision_model.clip_vision_model
                print(f"clip_vision_model类型: {type(clip_vision_model)}")
                print(f"clip_vision_model是否有encoder属性: {hasattr(clip_vision_model, 'encoder')}")
                
                if hasattr(clip_vision_model, 'encoder'):
                    encoder = clip_vision_model.encoder
                    print(f"encoder类型: {type(encoder)}")
                    print(f"encoder是否有layers属性: {hasattr(encoder, 'layers')}")
                    
                    if hasattr(encoder, 'layers'):
                        layers = encoder.layers
                        print(f"layers类型: {type(layers)}")
                        print(f"layers长度: {len(layers)}")
                        print("✅ 找到了transformer编码器层！")
                        return True
    
    print("❌ 未找到transformer编码器层")
    return False

if __name__ == "__main__":
    success = debug_model_structure()
    sys.exit(0 if success else 1)