from logging.handlers import RotatingFileHandler
from argparse import ArgumentParser
import logging
import os


class Logger:
    def __init__(self):
        self._logger = None

    @property
    def is_initialized(self):
        return self._logger is not None

    def info(self, *args, **kwargs):
        if self._logger is None:
            self.init_logger()
        self._logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        if self._logger is None:
            self.init_logger()
        self._logger.warn(*args, **kwargs)

    def init_logger(self, log_file=None, log_file_level=logging.NOTSET,
                    rotate=False, overwrite_log_file=False):
        log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.handlers = [console_handler]

        if log_file and log_file != '':
            if os.path.exists(log_file) and overwrite_log_file:
                os.remove(log_file)

            if rotate:
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=1000000, backupCount=10)
            else:
                file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_file_level)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)

        self._logger = logger


logger = Logger()