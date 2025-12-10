"""
Logging Utilities Module

Thread-safe logging utilities for multi-threaded processing.
"""

import logging
import logging.handlers
import queue
import sys
import os
from pathlib import Path
from typing import Optional

# Global logging state
_log_queue: queue.Queue = queue.Queue()
_log_listener: Optional[logging.handlers.QueueListener] = None
_logging_failed: bool = False


def setup_thread_safe_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup thread-safe logging with QueueHandler.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (placed in log_dir)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    global _log_listener, _logging_failed
    
    try:
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        except Exception as e:
            print(f"CRITICAL: Cannot setup console logging: {e}")
            _logging_failed = True
            raise RuntimeError(f"Console logging setup failed: {e}")
        
        # File handler with rotation (if specified)
        if log_file:
            try:
                # Create log directory
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                full_log_path = log_path / log_file
                
                # Create RotatingFileHandler
                file_handler = logging.handlers.RotatingFileHandler(
                    str(full_log_path),
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                
                # Create QueueHandler for thread safety
                queue_handler = logging.handlers.QueueHandler(_log_queue)
                logger.addHandler(queue_handler)
                
                # Create and start QueueListener
                _log_listener = logging.handlers.QueueListener(
                    _log_queue,
                    file_handler,
                    respect_handler_level=True
                )
                _log_listener.start()
                
                print(f"Thread-safe logging setup with file: {full_log_path}")
                
            except Exception as e:
                print(f"Warning: Could not create file handler: {e}")
                # Continue with console logging only
        
        return logger
        
    except Exception as e:
        print(f"CRITICAL: Logging system setup failed: {e}")
        _logging_failed = True
        raise RuntimeError(f"Logging system setup failed: {e}")


def stop_logging() -> None:
    """Stop the logging system and clean up."""
    global _log_listener, _logging_failed
    
    try:
        if _log_listener:
            _log_listener.stop()
            _log_listener = None
            print("Logging system stopped")
    except Exception as e:
        print(f"Warning: Error stopping logging system: {e}")
        _logging_failed = True


def safe_log(logger: logging.Logger, level: str, message: str) -> None:
    """
    Safe logging that falls back to print if logging fails.
    
    Args:
        logger: Logger instance
        level: Log level (INFO, WARNING, ERROR, CRITICAL)
        message: Message to log
    """
    global _logging_failed
    
    if _logging_failed:
        print(f"[{level.upper()}] {message}")
        return
    
    try:
        level_map = {
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "WARNING": logger.warning,
            "ERROR": logger.error,
            "CRITICAL": logger.critical
        }
        log_func = level_map.get(level.upper(), logger.info)
        log_func(message)
    except Exception as e:
        print(f"[{level.upper()}] {message}")
        print(f"Logging failed: {e}")
        _logging_failed = True


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary logging setup."""
    
    def __init__(
        self,
        log_file: str,
        level: str = "INFO",
        log_dir: str = "logs"
    ):
        self.log_file = log_file
        self.level = level
        self.log_dir = log_dir
        self.logger = None
    
    def __enter__(self) -> logging.Logger:
        self.logger = setup_thread_safe_logging(
            level=self.level,
            log_file=self.log_file,
            log_dir=self.log_dir
        )
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_logging()
        return False
