from .version import __version__, version_info
from .registry import Registry, MODELS, DATASETS, POSTFUNCS

from .datasets import * # Registry all Datasets to the DATASETS register
from .model_zoo import *
from .modules import *  

__all__ = ['__version__', 'version_info', 'MODELS', "DATASETS", "POSTFUNCS"]



