import shutil
from colorama import init, Fore, Style
from pathlib import Path

def copy_file(source: Path, destination: Path):
    """
    Copy a single file with path, with checking and asking about the existence. 
    """
    if not source.is_file():
        print(f"Error: {source} is not a file.")
        return

    if destination.exists():
        print(f"Warning: {destination.name} already exists in {destination.parent}.")
        user_input = input(f"Do you want to overwrite {destination.name}? (y/n): ").strip().lower()
        if user_input != 'y':
            print(f"Skipping {destination.name}.")
            return

    shutil.copy(source, destination)
    print(f"Copied {source.name} to {destination}.")

def copy_files(source: Path, destination):
    """
    Copy files under a path without recursion, with checking and asking about the existence. 
    """
    yes_to_all, none_to_all=False, False
    
    for template_file in source.iterdir():
        if template_file.is_file():
            if template_file.name == "__init__.py":
                continue  # skip __init__.py  files
            destination_file = Path(destination) / template_file.name
            if destination_file.exists():
                if none_to_all:
                    print(f'  Skipping {template_file.name}.\n')
                    continue
                if not yes_to_all:
                    # Alert , whether overwrite?
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