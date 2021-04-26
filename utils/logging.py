import logging

from .dist import is_master


logger_initialized = set([])


def get_logger(name, log_file: str=None, log_level: int=logging.INFO, file_mode: str='w') -> logging.Logger:
    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger

    logger_handlers = []
    logger_handlers.append(logging.StreamHandler())

    if is_master() and log_file is not None:
        logger_handlers.append(logging.FileHandler(log_file, file_mode))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    for handler in logger_handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_master():
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized.add(name)

    return logger
