import os
import argparse
import requests

from colorama import init, Fore, Style

# from IMDLBenCo.utils.paths import BencoPath
from IMDLBenCo.cli_funcs import cli_init, cli_env
import importlib.metadata 

PYPI_API_URL = 'https://pypi.org/pypi/IMDLBenCo/json'
from IMDLBenCo.version import __version__

def version_and_check_for_updates():
    print(f'IMDLBenCo codebase version: {__version__}')
    try:
        response = requests.get(PYPI_API_URL, timeout=5)
        response.raise_for_status()  # 如果响应码不是200，则抛出异常
        pypi_data = response.json()
        cloud_version = pypi_data['info']['version']  # 获取最新版本号
        # cloud_version = '0.1.21'   # for debug & test
        print("\tChecking for updates...")
        print("\tLocal version: ", __version__)
        print("\tPyPI newest version: ", cloud_version)
        local_version = __version__  # 通过 importlib.metadata 获取当前安装版本

        if cloud_version != local_version:
            print(Fore.YELLOW + f"New version available: {cloud_version}. Your version: {local_version}.")
            print("Run 'pip install --upgrade imdlbenco' to upgrade.")           
        else:
            print(Fore.GREEN + f"You are using the latest version: {local_version}.")
    except requests.exceptions.RequestException as e:
        print(Fore.RED + "Failed to check for updates from PyPI. Please check your internet connection.")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Command line interface for IMDLBenCo, with codebase version: ' + __version__)
    
    # Add version argument with update check only when user requests version
    parser.add_argument('-v', '--version', action='store_true', help="Show the version of the tool")
    
    subparsers = parser.add_subparsers(dest='command', required=False)
    
    # init command
    parser_init = subparsers.add_parser('init', help='Initialize the scripts and configs in a directory')
    init_subparsers = parser_init.add_subparsers(dest='subcommand', required=False)
    
    # init base
    parser_init_base = init_subparsers.add_parser('base', help='Initialize the base environment')
    parser_init_base.set_defaults(subcommand='base')

    # init model_zoo
    parser_init_model_zoo = init_subparsers.add_parser('model_zoo', help='Initialize the model zoo')
    
    # init backbone
    parser_init_backbone = init_subparsers.add_parser('backbone', help='Initialize the backbone')
    
    
    # env command
    parser_env = subparsers.add_parser('env', help='Show environment information')
    
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    if args.version:
        version_and_check_for_updates()
    
    if args.command == 'init':
        if args.subcommand is None:
            args.subcommand = 'base'
        cli_init(args.config, subcommand=args.subcommand)
        
    elif args.command == 'env':
        cli_env(args.config)

# def train(config):
#     print(f'Training with config: {config}')

# def evaluate(config):
#     print(f'Evaluating with config: {config}')

if __name__ == '__main__':
    main()
