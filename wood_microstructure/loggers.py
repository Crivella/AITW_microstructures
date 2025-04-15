import logging

logger: logging.Logger = None

formatter = logging.Formatter('{asctime} - {levelname:>7s} - {name:>6s}:{module:<15s} - {message}', style='{')

def add_file_logger(logger: logging.Logger, log_file_name: str):
    """Add a file logger to the logger."""
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def set_console_level(logger: logging.Logger, level: int):
    """Set the console level of the logger."""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)

def get_logger() -> logging.Logger:
    """Get the logger."""
    global logger
    if logger is not None:
        return logger

    logger = logging.getLogger('wood')
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
