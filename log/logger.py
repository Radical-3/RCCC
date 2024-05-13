import logging


class Logger:
    def __init__(self):
        self.__logger = logging.getLogger("logger")
        self.__logger.setLevel(logging.DEBUG)
        self.__formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.__level_dict = {"NOTSET": 0, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        self.__default_handler = logging.StreamHandler()
        self.__default_handler.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__default_handler)

    def set_config(self, config):
        self.__default_handler.close()
        self.__logger.removeHandler(self.__default_handler)

        self.__formatter = logging.Formatter(config.log_format)
        if config.enable_console_log:
            self.__enable_console(config.console_log_level)
            self.__logger.debug("enable console log")
        if config.enable_file_log:
            self.__enable_file(config.file_log_level, config.file_log_path)
            self.__logger.debug(f"enable file log,the log file is in the {config.file_log_path}")

    def __enable_console(self, level='DEBUG'):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.__level_dict[level])
        console_handler.setFormatter(self.__formatter)
        self.__logger.addHandler(console_handler)

    def __enable_file(self, level='DEBUG', log_file='./log/operating.log'):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.__level_dict[level])
        file_handler.setFormatter(self.__formatter)
        self.__logger.addHandler(file_handler)

    def debug(self, message):
        self.__logger.debug(message)

    def info(self, message):
        self.__logger.info(message)

    def warning(self, message):
        self.__logger.warning(message)

    def error(self, message):
        self.__logger.error(message)

    def critical(self, message):
        self.__logger.critical(message)


# 全局变量，用于在项目中获取全局日志对象
logger = Logger()
