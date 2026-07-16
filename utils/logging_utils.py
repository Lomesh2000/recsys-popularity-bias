"""Logging utilities."""

import os
import logging
from datetime import datetime
from typing import Optional
import torch


def setup_logger(
    name: str,
    log_dir: str = './logs',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{name}_{timestamp}.log"

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)

    return logger


def get_device(device_str: str = 'auto') -> torch.device:
    """Get torch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)
