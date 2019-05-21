import logging
from datetime import datetime
import os
import time

_logger = None
_log_path = None
_console_level = None

NOTSET = logging.NOTSET
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

def getLogger():
    global _logger, _console_level

    if _logger:
        return _logger

    logger = logging.getLogger('robamine')
    logger.setLevel(logging.DEBUG)
    fmt = Formatter(colors=True)
    console = logging.StreamHandler()
    if not _console_level:
        _console_level = logging.INFO
    console.setLevel(_console_level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    _logger = logger
    return logger

class Formatter(logging.Formatter):
    err_fmt = ('[%(name)s][%(levelname)s] %(message)s')
    warn_fmt = ('[%(name)s][%(levelname)s] %(message)s')
    err_fmt_colored = ('\033[1m\033[91m[%(name)s][%(levelname)s] %(message)s\033[0m')
    warn_fmt_colored = ('\033[1m\033[93m[%(name)s][%(levelname)s] %(message)s\033[0m')
    dbg_fmt  = "[%(name)s][%(levelname)s] %(message)s"
    info_fmt = ('[%(name)s][%(levelname)s] %(message)s')

    def __init__(self, colors=False):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')
        self.colors = colors

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = Formatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = Formatter.info_fmt

        elif record.levelno == logging.WARN:
            if self.colors:
                self._style._fmt = Formatter.warn_fmt_colored
            else:
                self._style._fmt = Formatter.warn_fmt


        elif record.levelno == logging.ERROR:
            if self.colors:
                self._style._fmt = Formatter.err_fmt_colored
            else:
                self._style._fmt = Formatter.err_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result

def init(directory='/tmp', console_level=logging.INFO, file_level=logging.DEBUG):
    global _logger, _log_path

    # Create the log path
    if not os.path.exists(directory):
        os.makedirs(directory)
    #log_path = os.path.join(directory, 'robamine_logger_' + agent_name.replace(" ", "_") + "_" + env_name.replace(" ", "_") + '_' + get_now_timestamp())
    log_path = os.path.join(directory, 'robamine_logs_' + get_now_timestamp())
    os.makedirs(log_path)
    _log_path = log_path

    logging.basicConfig(level=file_level,
                        format='[%(name)s][%(levelname)s] %(message)s',
                        filename=os.path.join(log_path, 'console.log'),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(Formatter(colors=True))
    logging.getLogger('robamine').addHandler(console)

    _logger = logging.getLogger('robamine')

    _logger.info('Logging in path: %s', log_path)
    _logger.info('Logging Console Level: %s, File Level: %s', logging.getLevelName(console_level), logging.getLevelName(file_level))

def get_logger_path():
    global _log_path

    if _log_path:
        return _log_path

    init()
    return _log_path

def get_now_timestamp():
    """
    Returns a timestamp for the current datetime as a string for using it in
    log file naming.
    """
    now_raw = datetime.now()
    return str(now_raw.year) + '.' + \
           '{:02d}'.format(now_raw.month) + '.' + \
           '{:02d}'.format(now_raw.day) + '.' + \
           '{:02d}'.format(now_raw.hour) + '.' \
           '{:02d}'.format(now_raw.minute) + '.' \
           '{:02d}'.format(now_raw.second) + '.' \
           '{:02d}'.format(now_raw.microsecond)
