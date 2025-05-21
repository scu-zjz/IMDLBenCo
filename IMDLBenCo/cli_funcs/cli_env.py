import os
import platform
from colorama import init, Fore, Style
from IMDLBenCo import __version__

import torch
import timm
import albumentations

def is_torch_cuda_available():
    """
    Check if CUDA is available for PyTorch.
    """
    return torch.cuda.is_available()

def get_env_info():
    info = {
        "`IMDLBenCo` version": __version__,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Torchvision version": torch.__version__,
        "TIMM version": timm.__version__,
        "Albumentations version": albumentations.__version__,
    }
    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()
        info["GPU number"] = torch.cuda.device_count()
        info["GPU memory"] = f"{torch.cuda.mem_get_info()[1] / (1024**3):.2f}GB"


    try:
        import deepspeed  # type: ignore

        info["DeepSpeed version"] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes  # type: ignore

        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass

    try:
        import vllm

        info["vLLM version"] = vllm.__version__
    except Exception:
        pass
    try:
        import subprocess
        # get the dir of imdlbenco package
        imdlbenco_dir = os.path.dirname(os.path.abspath(__file__))
        # move to this dir and get the git commit hash in a subprocess
        # but don't change the current working directory
        os.chdir(imdlbenco_dir)
        commit_info = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit_hash = commit_info.stdout.strip()
        info["Git commit"] = commit_hash
    except Exception:
        pass

    print("\n" + "\n".join([f"- {key}: {value}" for key, value in info.items()]) + "\n")


def cli_env(config):
    get_env_info()

if __name__ == "__main__":
    print(get_env_info())