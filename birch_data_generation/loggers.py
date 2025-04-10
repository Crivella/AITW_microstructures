import logging
import os

logger = None

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
    if log_file_name is not None:
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
