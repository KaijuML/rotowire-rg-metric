from logging.handlers import RotatingFileHandler
from itertools import zip_longest
from argparse import Namespace
import logging
import json
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


_container_sentinel = object()


class Container:
    """
    Dummy class that can be instantiated with arbitrary key-word arguments
    """
    def __init__(self, **kwargs):
        self.__setstate__(kwargs)

    def __setstate__(self, state):
        """Used by the python builtin multiprocessing library"""
        for key, value in state.items():
            setattr(self, key, value)

    def __getstate__(self):
        """Used by the python builtin multiprocessing library"""
        return self.__dict__

    def to_namespace(self):
        return Namespace(**self.__getstate__())

    def pop(self, item, default=_container_sentinel):
        if not hasattr(self, item):
            if default is not _container_sentinel:
                return default
            self._raise_unknown_attr(item)

        value = getattr(self, item)
        delattr(self, item)
        return value

    @staticmethod
    def _raise_unknown_attr(item):
        raise RuntimeError(f'<{item}> is not a known attribute of this Container.'
                           f' This code uses Containers at places where I need '
                           f'an object that behaves like another object (e.g. '
                           f'a batch, a namespace, etc.). Find where this '
                           f'container is used in the code and fix this issue!')

    def __getattr__(self, item):
        """Only called when item is not known to the container"""
        self._raise_unknown_attr(item)

    def __repr__(self):
        try:
            return f'Container({json.dumps(self.__getstate__(), indent=4)})'
        except TypeError:
            return f'Container(size={len(self.__getstate__())})'


def grouped(iterable, n):
    return zip_longest(*[iter(iterable)]*n)