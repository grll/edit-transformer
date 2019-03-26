import sys
import logging
import time
import os
from os.path import join, isdir
from os import makedirs


def setup_logging(env_variable_name="INTERACTIVE_ENVIRONMENT", root="/data/logs", force_file=False, lvl=logging.DEBUG,
                  noname=False):
    """Setup the basicConfig logging to file depending on a env_variable

    Args:
        env_variable_name (str): Name of the env variable to check for interactivity or not. If interactif -> log to
            console. Log to file otherwise.
        root (str): Path to the root folder where to put the logs.
        force_file (bool): weather to force output to file or not
        lvl (logging.level): level of logging requested.
        noname (bool): weather to use date directory and date in logfile or just `log.txt` if True.

    """

    fmt = '%(asctime)s | %(name)-60s | %(levelname)-8s | %(message)s'
    dfmt = '%d-%m %H:%M:%S'

    if noname:
        fpath = join(root, "log.txt")
    else:
        directory_path = join(root, str(time.strftime('%Y-%m-%d')))
        if not isdir(directory_path):
            makedirs(directory_path)
        fpath = join(directory_path, "log_" + str(time.strftime('%Y%m%d_%H%M%S')) + ".txt")

    def logging_to_console():
        logging.basicConfig(level=lvl, format=fmt, datefmt=dfmt)

    def logging_to_file():
        logging.basicConfig(level=lvl, format=fmt, datefmt=dfmt, filename=fpath, filemode="w")

        def exception_hook(exc_type, exc_value, exc_traceback):
            logging.error(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )

        sys.excepthook = exception_hook

    if env_variable_name in os.environ and force_file is False:
        logging_to_console()
    else:
        logging_to_file()
