import argparse
import os

from colorama import init, Fore, Style
from pathlib import Path
# from IMDLBenCo.utils.paths import BencoPath
from IMDLBenCo.cli_funcs import cli_init, cli_guide, cli_data

def main():
    parser = argparse.ArgumentParser(description='Command line interface for IMDLBenCo')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    
    # init command
    parser_init = subparsers.add_parser('init', help='Initialize the environment')
    init_subparsers = parser_init.add_subparsers(dest='subcommand', required=False)
    
    # init base
    parser_init_base = init_subparsers.add_parser('base', help='Initialize the base environment')
    parser_init_base.set_defaults(subcommand='base')

    # init model_zoo
    parser_init_model_zoo = init_subparsers.add_parser('model_zoo', help='Initialize the model zoo')
    
    # init backbone
    parser_init_backbone = init_subparsers.add_parser('backbone', help='Initialize the backbone')
    
    
    # guide command
    parser_guide = subparsers.add_parser('guide', help='Guide for using the tool')
    
    # data command
    parser_data = subparsers.add_parser('data', help='Manage data')
    
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    
    if args.command == 'init':
        if args.subcommand is None:
            args.subcommand = 'base'
        cli_init(args.config, subcommand=args.subcommand)
        
    elif args.command == 'guide':
        cli_guide(args.config)
    elif args.command == 'data':
        cli_data(args.config)

def train(config):
    print(f'Training with config: {config}')

def evaluate(config):
    print(f'Evaluating with config: {config}')

if __name__ == '__main__':
    main()
