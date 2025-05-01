import logging
import sys
from collections.abc import Callable
from typing import Dict, Generator, List, Optional, Tuple, Type, Union, Any

import difflib
from rich.console import Console
from rich.table import Table

from .utils.misc import is_seq_of

class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.
    
    """
    def __init__(self,
                name: str,
                # build_func: Optional[Callable] = None,
                # parent: Optional['Registry'] = None,   # Using strings to address circular import issues                
                ):
        # from .build_functions import build_from_cfg
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        # self._children: Dict[str, 'Registry'] = dict()
        # self._imported = False
        
        # self.parent: Optional['Registry']
        
        # if parent is not None:
        #     assert isinstance(parent, Registry)
        #     parent._add_child(self)
        #     self.parent = parent
        # else:
        #     self.parent = None

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 3. build_from_cfg
        # self.build_func: Callable
        # if build_func is None:
            # self.build_func = build_from_cfg
        # else:
            # self.build_func = build_func
            
    def __len__(self):
        return len(self._module_dict)
    
    def __contains__(self, key):
        return self.get(key) is not None
    
    
    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()
    
    @property
    def name(self):
        return self._name

    def has(self, name: str) -> bool:
        """Check if a name is in the registry."""
        return name in self._module_dict

    @property
    def module_dict(self):
        return self._module_dict
    
    def _suggest_correction(self, input_string: str) -> Optional[str]:
        """Suggest the most similar string from the registered modules."""
        suggestions = difflib.get_close_matches(input_string, self._module_dict.keys(), n=3, cutoff=0.3)
        if suggestions:
            return suggestions[0]
        return None
    
    def get(self, name):
        if name in self._module_dict:
            return self._module_dict[name]
        suggestion = self._suggest_correction(name)
        print(f"{self}")
        if suggestion:
            raise KeyError(f'"{name}" is not registered in {self.name}. Did you mean "{suggestion}"?')
        else:
            raise KeyError(f'"{name}" is not registered in {self.name} and no similar names were found.')
    def get_lower(self, name):
        """Get a module by name, ignoring case."""
        for key in self._module_dict.keys():
            if key.lower() == name.lower():
                return self._module_dict[key]
        suggestion = self._suggest_correction(name)
        lower_name = name.lower()
        print(f"{self}")
        if suggestion:
            lower_suggestion = suggestion.lower()
            raise KeyError(f'Nether "{name}" nor lower-case "{lower_name}" is registered in {self.name}. Did you mean "{suggestion}" or "{lower_suggestion}"?')
        else:
            raise KeyError(f'Nether "{name}" nor lower-case "{lower_name}" is registered in {self.name} and no similar names were found.')

    # @property
    # def children(self):
    #     return self._children

    # @property
    # def root(self):
    #     return self._get_root_registry()
    
    # def _get_root_registry(self) -> 'Registry':
    #     """Return the root registry."""
    #     root = self
    #     while root.parent is not None:
    #         root = root.parent
    #     return root

    
    def _register_module(self,
                        module: Type,
                        module_name: Optional[Union[str, List[str]]] = None,
                        force: bool = False) -> None:
        """Register a module.

        Args:
            module (type): Module to be registered. Typically a class or a
                function, but generally all ``Callable`` are acceptable.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                            f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            force: bool = False,
            module: Optional[Type] = None) -> Union[type, Callable]:
        """Register a module.

        A record will be added to ``self._module_dict``, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Args:
            name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
            module (type, optional): Module class or function to be registered.
                Defaults to None.

        Examples:
            >>> backbones = Registry('backbone')
            >>> # as a decorator
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> # as a normal function
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(module=ResNet)
        """

        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register
    
    
    def build(self, name: dict, *args, **kwargs) -> Any:
        """Build an instance.

        Build an instance by calling :attr:`build_func`.

        Args:
            cfg (dict): Config dict needs to be built.

        Returns:
            Any: The constructed object.

        Examples:
            >>> from mmengine import Registry
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     def __init__(self, depth, stages=4):
            >>>         self.depth = depth
            >>>         self.stages = stages
            >>> cfg = dict(type='ResNet', depth=50)
            >>> model = MODELS.build(cfg)
        """
        return self.get(name)(*args, **kwargs)
        # return self.build_func(cfg, *args, **kwargs, registry=self)
    

MODELS = Registry(name = 'MODELS')

DATASETS = Registry(name = 'DATASETS')

POSTFUNCS = Registry(name = 'POSTFUNCS')
