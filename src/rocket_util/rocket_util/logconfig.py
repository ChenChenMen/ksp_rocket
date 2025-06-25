"""Define common utility functions for ksp rocket."""

import logging


class CustomFormatter(logging.Formatter):
    """Custom formatter to add hardcoded colors based on log levels."""

    # Define log level colors (ANSI escape codes)
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35;1m",  # Bright Magenta
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record):
        # Apply color to the level name
        levelname_color = self.COLORS.get(record.levelno, "")
        record.msg = f"{levelname_color}{record.msg}{self.RESET}"

        # Format the message
        return super().format(record)


def create_logger(name, level=logging.DEBUG, log_file=None):
    """
    Create a custom logger with hardcoded colored output for each log level.

    Args:
        name (str): Name of the logger, typically `__name__`.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file (str, optional): File to log messages (in addition to console).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Define custom log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = CustomFormatter(log_format)

    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))  # No colors for file
        logger.addHandler(file_handler)

    return logger
