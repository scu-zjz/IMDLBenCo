import os
import re
from colorama import init, Fore, Style
from IMDLBenCo.utils.paths import BencoPath
from .copy_funcs import copy_files, copy_file 

def _copy_train_scripts():
    current_dir = os.getcwd()
    # Copy train scripts
    target_dir = os.path.join(current_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files(BencoPath.get_templates_dir(), target_dir)
    
    
def _copy_dataset_json():
    current_dir = os.getcwd()
    # Copy train scripts
    target_dir = os.path.join(current_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files(BencoPath.get_dataset_json_dir(), target_dir)
    
def _copy_init_base_files():
    current_dir = os.getcwd()
    # Copy train scripts
    target_dir = os.path.join(current_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files(BencoPath.get_init_base_dir(), target_dir)
    
def _copy_demo_runs():
    # Copy demo runs
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, 'runs')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(os.path.join(target_dir, "test_save_images")):
        os.makedirs(os.path.join(target_dir, "test_save_images"))
    if not os.path.exists(os.path.join(target_dir, "test_complexity")):
        os.makedirs(os.path.join(target_dir, "test_complexity"))
    copy_files(BencoPath.get_model_zoo_runs_dir(), target_dir)
    copy_files(BencoPath.get_model_zoo_runs_dir() / "test_save_images" , os.path.join(target_dir, "test_save_images"))
    copy_files(BencoPath.get_model_zoo_runs_dir() / "test_complexity" , os.path.join(target_dir, "test_complexity"))

    
def _copy_demo_configs():
    # Copy demo configs
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, 'configs')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files(BencoPath.get_model_zoo_configs_dir(), target_dir)
    
# hejack train.py test.py test_robust.py with self defined model
def _inject_after_last_import(path, inject):
    # 读取文件内容
    with open(path, 'r') as file:
        content = file.readlines()
    
    base_file_name = os.path.basename(path)
    # 检查是否已经存在指定的导入语句
    if any(inject in line for line in content):
        print(f"  The specified import statement already exists in {base_file_name}.")
        return
    
    # 正则表达式匹配 from ... import ... 行
    pattern = re.compile(r'^\s*from\s+\S+\s+import\s+(?!.*\()(\S+(,\s*\S+)*)\s*$')
    
    # 找到最后一个匹配的行的索引
    last_import_index = -1
    for i, line in enumerate(content):
        if pattern.match(line):
            last_import_index = i
    
    # 如果找到匹配的行，则在其后插入新的内容
    if last_import_index != -1:
        content.insert(last_import_index + 1, inject + '\n')
    
    # re-save the file
    with open(path, 'w') as file:
        file.writelines(content)
    print(f"  Injected '{inject}' into {path}.")

def cli_init(config, subcommand):
    print(f'{Fore.GREEN}Initializing in current working directory...{Style.RESET_ALL}')
    
    # base initialize that only contain default scripts
    if subcommand == "base":
        _copy_train_scripts()
        _copy_init_base_files()
        _copy_dataset_json()
        for name in ['train.py', 'test.py', 'test_robust.py', 'test_complexity.py', 'test_save_images.py']:
            injected_str = "from mymodel import MyModel  # TODO, you need to change this line when modifying the name model"
            current_dir = os.getcwd()
            target_dir = os.path.join(current_dir, name)
            _inject_after_last_import(target_dir, injected_str)
        
    if subcommand == "model_zoo":
        _copy_train_scripts()
        _copy_demo_runs() 
        _copy_demo_configs()
        _copy_dataset_json()
    # base initialize that only contain default scripts
    if subcommand == "backbone":
        _copy_train_scripts()
        _copy_demo_runs() 
        _copy_demo_configs()
        _copy_dataset_json()
    print(f'{Fore.GREEN}Successfully initialized IMDLBenCo scripts.{Style.RESET_ALL}')