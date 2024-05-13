from .version import __version__, version_info
from .registry import Registry

__all__ = ['__version__', 'version_info']


MODELS = Registry(name = 'models')

DATASETS = Registry(name = 'datasets')

PROTOCOLS = Registry(name = 'protocols')

