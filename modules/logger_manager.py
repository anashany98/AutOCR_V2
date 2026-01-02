"""
Unified logging for AutOCR.

This module sets up a standard Python ``logging`` configuration that logs
messages both to the console and to a rotating log file.  In addition it
provides a custom ``logging.Handler`` implementation which persists log
entries to the SQL database via the ``DBManager``.  The result is that
calling ``logger.info(...)`` writes simultaneously to the terminal,
``postbatch.log`` and the ``logs`` table.

Usage:

>>> from modules.db_manager import DBManager
>>> from modules.logger_manager import setup_logger
>>> db = DBManager(db_path='my.db')
>>> logger = setup_logger('postbatch.log', 'INFO', db)
>>> logger.info('Processing started')

The log level is configurable; by default it uses INFO.  Exception traces
are included in the log file when ``exc_info=True`` is passed to the log
call.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from .db_manager import DBManager


class DBLogHandler(logging.Handler):
    """Custom log handler that writes events to the database via DBManager."""

    def __init__(self, db_manager: DBManager) -> None:
        super().__init__()
        self.db_manager = db_manager

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            detail = None
            if record.exc_info:
                # Format the exception traceback separately
                detail = self.formatException(record.exc_info)
            self.db_manager.insert_log(
                event=msg,
                detail=detail,
                level=record.levelname,
            )
        except Exception:
            # Avoid infinite recursion if logging from within DB insert
            pass


def setup_logger(
    log_file: str,
    level: str = "INFO",
    db_manager: Optional[DBManager] = None,
) -> logging.Logger:
    """
    Configure and return a logger instance for the AutOCR application.

    Parameters
    ----------
    log_file:
        Path to a log file.  A rotating handler will write up to 5 MB per
        file and keep up to 5 backups.
    level:
        The minimum severity of messages to emit.  Defaults to ``INFO``.
    db_manager:
        Optional DBManager instance.  If provided, a database log handler
        will be attached which writes each log record into the database.

    Returns
    -------
    logging.Logger
        Configured logger ready for use.
    """
    logger = logging.getLogger("AutOCR")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # Do not pass to root logger

    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Clear any existing handlers attached to this logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    # Optional database handler
    if db_manager is not None:
        db_handler = DBLogHandler(db_manager)
        db_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(db_handler)

    return logger