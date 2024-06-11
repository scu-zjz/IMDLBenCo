import argparse
import inspect
import sys
sys.path.append("./")
from IMDLBenCo import MODELS

def create_argparser(model_class):
    parser = argparse.ArgumentParser(description=f"Arguments for {model_class.__name__}")
    
    # 获取模型的__init__方法的签名
    sig = inspect.signature(model_class.__init__)
    
    # 解析每个参数并添加到argparse
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        arg_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        default_value = param.default if param.default != inspect.Parameter.empty else None
        print(name, arg_type, default_value)
        
        if default_value is not None:
            parser.add_argument(f'--{name}', type=arg_type, default=default_value, help=f'{name} (default: {default_value})')
        else:
            parser.add_argument(f'--{name}', type=arg_type, required=True, help=f'{name} (required)')
    
    return parser


iml_vit = MODELS.get("IML_ViT")
create_argparser(iml_vit)