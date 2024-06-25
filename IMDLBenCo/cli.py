import argparse
import os
import shutil
from colorama import init, Fore, Style
from pathlib import Path
from IMDLBenCo.utils.paths import BencoPath


def copy_files(source: Path, destination):
    yes_to_all, none_to_all=False, False
    
    for template_file in source.iterdir():
        if template_file.is_file():
            if template_file.name == "__init__.py":
                continue  # 跳过 __init__.py 文件
            destination_file = Path(destination) / template_file.name
            if destination_file.exists():
                if none_to_all:
                    print(f'  Skipping {template_file.name}.\n')
                    continue
                if not yes_to_all:
                    # 提示用户确认覆盖
                    print(f'  {Fore.YELLOW}Warning: {template_file.name} already exists in {destination}.{Style.RESET_ALL}')
                    user_input = input(f'  Do you want to overwrite {template_file.name}? (y/n/all/none): ').strip().lower()
                    if user_input == 'all':
                        yes_to_all = True
                    elif user_input == 'none':
                        none_to_all = True
                        print(f'  Skipping {template_file.name}.\n')
                        continue
                    elif user_input != 'y':
                        print(f'  Skipping {template_file.name}.\n')
                        continue
            shutil.copy(template_file, destination_file)
            print(f'  Copied {template_file.name} to {destination}.\n')

# def copy_template_files(destination):
#     # 获取当前文件所在目录
#     current_dir = BencoPath.get_package_dir()
#     # print(current_dir)
#     templates_dir = BencoPath.get_templates_dir()

#     # 列出所有模板文件并复制到目标目录
#     for template_file in templates_dir.iterdir():
#         if template_file.is_file():
#             destination_file = Path(destination) / template_file.name
#             if destination_file.exists():
#                 # 提示用户确认覆盖
#                 print(f'  {Fore.YELLOW}Warning: {template_file.name} already exists in {destination}.{Style.RESET_ALL}')
#                 user_input = input(f'  Do you want to overwrite {template_file.name}? (y/n): ').strip().lower()
#                 if user_input != 'y':
#                     print(f'  Skipping {template_file.name}.\n')
#                     continue
#             shutil.copy(template_file, destination_file)
#             print(f'  Copied {template_file.name} to {destination}.\n')

def main():    
    parser = argparse.ArgumentParser(description='Command line for IMDLBenCo')
    parser.add_argument(
        'command', 
        choices=[
            'init', 
            'guide', # TODO
            'data'   # TODO
        ], 
        help='Command to execute'
    )
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init(args.config)
    elif args.command == 'guide':
        train(args.config)
    elif args.command == 'data':
        evaluate(args.config)

def init(config):
    print(f'{Fore.GREEN}Initializing in current working directory...{Style.RESET_ALL}')
    current_dir = os.getcwd()
    # Copy train scripts
    target_dir = os.path.join(current_dir, 'IMDLBenCo', 'training_scripts')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files(BencoPath.get_templates_dir(), target_dir)
    
    # Copy demo runs
    target_dir = os.path.join(current_dir, 'runs')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files(BencoPath.get_runs_dir(), target_dir)
    
    # Copy demo configs
    target_dir = os.path.join(current_dir,'IMDLBenCo', 'configs')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files(BencoPath.get_configs_dir(), target_dir)
    
    
    
    print(f'{Fore.GREEN}Successfully initialized IMDLBenCo scripts.{Style.RESET_ALL}')
def train(config):
    print(f'Training with config: {config}')

def evaluate(config):
    print(f'Evaluating with config: {config}')

if __name__ == '__main__':
    main()
