import types
import yaml


class Config:
    def __init__(self, logger=None, config_path='./config/base.yaml'):
        self.__config = None
        with open(config_path, 'r', encoding='utf-8') as f:
            self.__config = yaml.safe_load(f)
            self.__config = types.SimpleNamespace(**self.__config)
        if logger is not None:
            logger.info(f"Loaded config from {config_path},with {len(vars(self.__config))} parameters")

    def item(self):
        return self.__config
