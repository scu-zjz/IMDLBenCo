
import os
from pathlib import Path
import appdirs

class BencoPath:
    app_name = "IMDLBenCo"
    app_author = "MyCompany"

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
    def get_runs_dir(cls):
        return cls.get_package_dir().parent / 'runs'