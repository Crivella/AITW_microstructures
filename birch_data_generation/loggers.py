import logging
import os

logger: logging.Logger = None

def add_file_logger(logger: logging.Logger, log_file_name):
    """Add a file logger to the logger."""
    # Formatter
    formatter = logging.Formatter('{asctime} - {levelname:>7s} - {name:>6s}:{module:<15s} - {message}', style='{')

    # File handler
    if log_file_name is not None:
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def set_console_level(logger: logging.Logger, level: int):
    """Set the console level of the logger."""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)

if logger is None:
    # Create a logger
    logger = logging.getLogger('wood')
    logger.setLevel(logging.DEBUG)

    # Formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('{asctime} - {levelname:>7s} - {name:>6s}:{module:<15s} - {message}', style='{')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file_name = os.getenv('LOG_FILE_NAME', None)
    add_file_logger(logger, log_file_name)
