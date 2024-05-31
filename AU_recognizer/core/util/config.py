import configparser
import logging.config
import platform
from datetime import date

from AU_recognizer.core.util.constants import *


def __get_log_name():
    return date.today().strftime("%Y_%m_%d.log")


def purge_option_config(option):
    logger.info("project: " + option + " does not exist or is not valid or is closed, remove it from open projects")
    nect_config.remove_option(OPEN_PROJECTS, option)
    write_config()


def write_config():
    logger.debug(f"save pynect configuration file")
    with open(CONFIG_FILE, 'w') as config_file:
        nect_config.write(config_file)


# load config file
nect_config = configparser.ConfigParser()
nect_config.optionxform = str
exist = nect_config.read(CONFIG_FILE)
if not exist:
    nect_config[CONFIG] = {
        LANGUAGE: LANGUAGE_DEFAULT,
        I18N_PATH: I18N_PATH_DEFAULT,
        LOGGER: LOGGER_DEFAULT,
        LOGGER_PATH: LOGGER_PATH_DEFAULT,
        LOG_FOLDER: LOG_FOLDER_DEFAULT,
        PROJECTS_FOLDER: PROJECTS_FOLDER_DEFAULT,
        MODEL_FOLDER: MODEL_FOLDER_DEFAULT,
        GUIDE_FILE: GUIDE_FILE_DEFAULT
    }
    nect_config[OPEN_PROJECTS] = {}
# global logger
logging.config.fileConfig(fname=Path(nect_config[CONFIG][LOGGER_PATH]), disable_existing_loggers=False,
                          defaults={
                              LOG_FOLDER: str((Path(nect_config[CONFIG][LOG_FOLDER]) / __get_log_name()).resolve())})
logger = logging.getLogger(nect_config[CONFIG][LOGGER])
if not exist:
    write_config()

operating_system = platform.system()

windowing_system = ""

logger.debug("config.py imported")
