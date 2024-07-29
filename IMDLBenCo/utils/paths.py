
import os
from pathlib import Path
import appdirs

class BencoPath:
    app_name = "IMDLBenCo"
    app_author = "IMDLBenCo-authors"

    @classmethod
    def get_data_storage_path(cls):
        storage_path = appdirs.user_data_dir(cls.app_name, cls.app_author)
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        return storage_path

    @classmethod
    def get_package_dir(cls):
        return Path(__file__).parent.parent

    @classmethod
    def get_templates_dir(cls):
        return cls.get_package_dir() / 'training_scripts'
    
    @classmethod
    def get_dataset_json_dir(cls):
        return cls.get_package_dir() / 'statics' / 'dataset_json'

    @classmethod
    def get_init_base_dir(cls):
        return cls.get_package_dir() / 'statics' / 'base'

    @classmethod
    def get_model_zoo_runs_dir(cls):
        return cls.get_package_dir() / 'statics' / 'model_zoo' / 'runs'
    
    @classmethod
    def get_model_zoo_configs_dir(cls):
        return cls.get_package_dir() / 'statics' / 'model_zoo' / 'configs'