"""
Structured logging utility for the recommendation system.
Provides consistent logging across all modules with file and console output.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """
    Centralized logging manager for the recommendation system.
    
    Features:
    - Console and file output
    - Structured log format
    - Log rotation support
    - Module-specific loggers
    - Configurable log levels
    """
    
    _loggers: dict = {}
    _default_level = logging.INFO
    _log_dir: Optional[Path] = None
    
    @classmethod
    def set_log_directory(cls, log_dir: Path) -> None:
        """Set the directory for log files."""
        cls._log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_logger(
        cls, 
        name: str, 
        level: Optional[int] = None,
        log_to_file: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name (typically __name__ of the module)
            level: Logging level (default: INFO)
            log_to_file: Whether to write logs to file
        
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level or cls._default_level)
        
        # Avoid adding handlers multiple times
        if not logger.handlers:
            # Create formatter
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level or cls._default_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler (optional)
            if log_to_file and cls._log_dir:
                log_file = cls._log_dir / f"{name}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level or cls._default_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger


def get_logger(
    name: str, 
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_to_file: Whether to write to file
    
    Returns:
        Configured logger
    """
    from shared.constants import Constants
    
    # Initialize log directory if needed
    if Logger._log_dir is None:
        Logger.set_log_directory(Constants.LOGS_DIR)
    
    return Logger.get_logger(name, level, log_to_file)


class LogContext:
    """
    Context manager for structured logging with additional context.
    
    Usage:
        with LogContext(logger, 'operation_name', extra_info={'key': 'value'}):
            # operation code
    """
    
    def __init__(
        self, 
        logger: logging.Logger, 
        operation: str,
        extra_info: Optional[dict] = None
    ):
        self.logger = logger
        self.operation = operation
        self.extra_info = extra_info or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(
            f"Starting {self.operation}",
            extra={**self.extra_info, 'status': 'started'}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {duration:.2f}s",
                extra={**self.extra_info, 'status': 'success', 'duration': duration}
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration:.2f}s: {exc_val}",
                extra={**self.extra_info, 'status': 'failed', 'error': str(exc_val)}
            )
        return False  # Don't suppress exceptions


class PerformanceLogger:
    """
    Decorator for logging function performance.
    
    Usage:
        @PerformanceLogger.log_performance(logger)
        def my_function():
            pass
    """
    
    @staticmethod
    def log_performance(logger: logging.Logger):
        """Decorator to log function execution time."""
        def decorator(func):
            import functools
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = datetime.now()
                logger.info(f"Starting {func.__name__}")
                try:
                    result = func(*args, **kwargs)
                    duration = (datetime.now() - start).total_seconds()
                    logger.info(f"Completed {func.__name__} in {duration:.2f}s")
                    return result
                except Exception as e:
                    duration = (datetime.now() - start).total_seconds()
                    logger.error(f"Failed {func.__name__} after {duration:.2f}s: {e}")
                    raise
            return wrapper
        return decorator
