"""
Logger utilities for C2Net EDL project.

This module provides logging functionality compatible with the original C2Net codebase.
"""

import logging
import os
import sys


def create_logger(name, log_file=None, level=logging.INFO):
    """
    Create a logger with the specified name and optional log file.

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level

    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger
_logger = create_logger(__name__)


def get_logger(name=None):
    """Get a logger instance."""
    if name is None:
        return _logger
    return create_logger(name)


if __name__ == '__main__':
    # Test logger
    logger = get_logger('test_logger')
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")