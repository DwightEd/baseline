"""Logging utilities."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    colored: bool = True,
) -> None:
    """Configure logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper()))
    
    if json_format:
        console.setFormatter(JsonFormatter())
    elif colored and sys.stdout.isatty():
        console.setFormatter(ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
    else:
        console.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
    root.addHandler(console)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter() if json_format else logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s'
        ))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


class ProgressLogger:
    """Progress logging for long operations."""
    def __init__(self, logger: logging.Logger, total: int, prefix: str = "Progress", interval: int = 10):
        self.logger = logger
        self.total = total
        self.prefix = prefix
        self.interval = interval
        self.current = 0
        self._last_percent = -1
    
    def update(self, n: int = 1) -> None:
        self.current += n
        percent = int(100 * self.current / self.total) if self.total > 0 else 100
        if percent != self._last_percent and percent % self.interval == 0:
            self.logger.info(f"{self.prefix}: {self.current}/{self.total} ({percent}%)")
            self._last_percent = percent
    
    def finish(self) -> None:
        self.logger.info(f"{self.prefix}: Completed {self.current}/{self.total}")
    
    def __enter__(self) -> 'ProgressLogger':
        return self
    
    def __exit__(self, *args) -> None:
        self.finish()


class LogContext:
    """Context manager for operation logging."""
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self) -> 'LogContext':
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time
        elapsed = time.time() - self.start_time
        if exc_type:
            self.logger.error(f"Failed: {self.operation} ({elapsed:.2f}s) - {exc_val}")
        else:
            self.logger.info(f"Completed: {self.operation} ({elapsed:.2f}s)")
