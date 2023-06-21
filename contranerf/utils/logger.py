import os
import sys
import atexit
import logging
from termcolor import colored
from iopath.common.file_io import PathManager

__all__ = ["setup_logger", ]

path_manager = PathManager()


class _ColorfulFormatter(logging.Formatter):
    def formatMessage(self, record):
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(output=None, distributed_rank=0, *, color=True, name="nerf"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
    )

    # create stdout handler
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            ch_formatter = _ColorfulFormatter(
                fmt=colored('[%(asctime)s %(name)s]: ', 'cyan') + '%(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
            )
        else:
            ch_formatter = formatter
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    # create file handler
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            filename = output
        else:
            filename = os.path.join(output, 'log.txt')
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        path_manager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = path_manager.open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io
