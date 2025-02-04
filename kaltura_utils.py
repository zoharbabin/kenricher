#!/usr/bin/env python
"""
kaltura_utils.py - Utility module for Kaltura client management and custom logging.
"""

import time
import logging
import functools
from typing import Optional, Callable, Tuple, Type

import coloredlogs
from coloredlogs import ColoredFormatter

from KalturaClient.Base import IKalturaLogger, KalturaConfiguration
from KalturaClient.Plugins.Core import KalturaSessionType
from KalturaClient import KalturaClient


class CustomFormatter(ColoredFormatter):
    """
    A custom log message formatter that supports colorizing messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified log record with timestamp, file, function, and line number.
        """
        color = getattr(record, "color", "white").lower()
        record.levelname = coloredlogs.ansi_wrap(record.levelname, color=color)
        record.msg = coloredlogs.ansi_wrap(record.msg, color=color)
        return super(CustomFormatter, self).format(record)


def create_custom_logger(logger: logging.Logger, file_path: Optional[str] = None) -> logging.Logger:
    """
    Configures the specified logger to use a custom handler and formatter.

    The log format now includes time, filename, function name, and line number.
    """
    # Clear existing handlers for a clean configuration.
    logging.getLogger().handlers.clear()
    logger.handlers.clear()
    handler: logging.Handler
    if file_path:
        handler = logging.FileHandler(file_path, encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    formatter = CustomFormatter(
        "%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(levelname)s:%(name)s:%(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def retry_on_exception(max_retries: int = 3, delay: float = 1, backoff: float = 2,
                       exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Callable:
    """
    Decorator for retrying a function call in case of specified exceptions.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_retries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as error:
                    msg = f"{error}, Retrying in {mdelay} seconds..."
                    logging.critical("Retrying function due to error: %s", msg, extra={"color": "red"})
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)  # Final attempt.
        return wrapper
    return decorator


class KalturaClientsManager:
    """
    Manager for Kaltura clients, providing both admin and user session clients.
    """

    def __init__(self, should_log: bool, kaltura_user_id: str, session_duration: int,
                 session_privileges: str, source_client_params: dict, dest_client_params: dict) -> None:
        """
        Initialize the KalturaClientsManager with separate clients for source and destination.
        """
        self.source_client_params = source_client_params
        self.dest_client_params = dest_client_params
        self.default_session_duration = session_duration
        self.default_session_privileges = session_privileges

        # Create source and destination clients (both ADMIN and USER sessions).
        self._source_client = self._create_kaltura_client(
            should_log, KalturaSessionType.ADMIN, kaltura_user_id, source_client_params
        )
        self._source_user_client = self._create_kaltura_client(
            should_log, KalturaSessionType.USER, kaltura_user_id, source_client_params
        )
        self._dest_client = self._create_kaltura_client(
            should_log, KalturaSessionType.ADMIN, kaltura_user_id, dest_client_params
        )
        self._dest_user_client = self._create_kaltura_client(
            should_log, KalturaSessionType.USER, kaltura_user_id, dest_client_params
        )

    @property
    def source_client(self) -> KalturaClient:
        """
        Get the source client (ADMIN session).
        """
        if not self._source_client:
            raise ValueError("source_client has not been initialized")
        return self._source_client

    @property
    def dest_client(self) -> KalturaClient:
        """
        Get the destination client (ADMIN session).
        """
        if not self._dest_client:
            raise ValueError("dest_client has not been initialized")
        return self._dest_client

    def get_source_client_with_user_session(self, user_id: str,
                                              session_duration: Optional[int] = None,
                                              session_privileges: Optional[str] = None) -> KalturaClient:
        """
        Creates a new USER session for the source client.
        """
        duration = session_duration or self.default_session_duration
        privileges = session_privileges or self.default_session_privileges
        return self._set_client_session(self._source_user_client, KalturaSessionType.USER, user_id,
                                        self.source_client_params["partner_id"],
                                        self.source_client_params["partner_secret"],
                                        duration, privileges)

    def get_dest_client_with_user_session(self, user_id: str,
                                            session_duration: Optional[int] = None,
                                            session_privileges: Optional[str] = None) -> KalturaClient:
        """
        Creates a new USER session for the destination client.
        """
        duration = session_duration or self.default_session_duration
        privileges = session_privileges or self.default_session_privileges
        return self._set_client_session(self._dest_user_client, KalturaSessionType.USER, user_id,
                                        self.dest_client_params["partner_id"],
                                        self.dest_client_params["partner_secret"],
                                        duration, privileges)

    def _set_client_session(self, client: KalturaClient, session_type: int, user_id: str,
                            partner_id: int, partner_secret: str,
                            session_duration: int, session_privileges: str) -> KalturaClient:
        """
        Sets a new KS on a Kaltura client instance.
        """
        ks = client.generateSessionV2(
            partner_secret,
            user_id,
            session_type,
            partner_id,
            session_duration,
            session_privileges
        )
        client.setKs(ks)  # type: ignore
        return client

    def _create_kaltura_client(self, should_log: bool, session_type: int, user_id: str,
                           client_params: dict) -> KalturaClient:
        """
        Instantiate a Kaltura client with a valid session.
        """
        class KalturaLogger(IKalturaLogger):
            def __init__(self):
                # Clear existing log file content.
                with open("kaltura_log.txt", "w", encoding="utf-8"):
                    pass
                # Create a custom logger that logs to both console and file.
                self.logger = create_custom_logger(logging.getLogger("kaltura_client"), "kaltura_log.txt")
                # Force the Kaltura logger to DEBUG level.
                self.logger.setLevel(logging.DEBUG)
            
            def log(self, msg: str) -> None:
                # Log at DEBUG level to capture detailed API call information.
                self.logger.debug(msg)

        config = KalturaConfiguration()
        config.serviceUrl = client_params["service_url"]
        if should_log:
            config.setLogger(KalturaLogger())
        client = KalturaClient(config, True)
        return self._set_client_session(
            client, session_type, user_id,
            client_params["partner_id"], client_params["partner_secret"],
            self.default_session_duration, self.default_session_privileges
        )

