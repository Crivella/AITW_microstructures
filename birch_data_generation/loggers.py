import logging

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
    file_handler = logging.FileHandler('birch_data_generation.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
