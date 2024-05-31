from configparser import ConfigParser

from AU_recognizer.core.util import nect_config
from AU_recognizer.core.util.config import write_config, logger
from AU_recognizer.core.util.constants import *


def create_project_folder(path: Path):
    try:
        path.mkdir()
        (path / F_OUTPUT).mkdir()
        (path / F_INPUT).mkdir()
    except (FileExistsError, FileNotFoundError):
        logger.exception("Creation of the directories failed")
    else:
        logger.debug("Successfully created the project directories")


def add_to_open_projects(path: Path, name=""):
    logger.debug(f"add project {path} to open projects in config file")
    if name:
        nect_config.set(OPEN_PROJECTS, str(path / name), name)
    else:
        nect_config.set(OPEN_PROJECTS, str(path), path.name)
    write_config()


def store_open_project(path, name):
    logger.debug(f"create project {name}.ini config file")
    p_config = ConfigParser()
    p_config.add_section(name)
    p_config.set(name, P_NAME, name)
    p_config.set(name, P_PATH, str(path / name))
    with open((path / name / (name + '.ini')), 'w') as f:
        p_config.write(f)
    add_to_open_projects(path, name)
    return p_config


def rename_project(config_path: Path, old_name, new_name):
    logger.debug(f"rename project {old_name} to {new_name}")
    p_config = ConfigParser()
    exist = p_config.read(config_path)
    if exist:
        rename_section(p_config, old_name, new_name)
        p_config.set(new_name, P_NAME, new_name)
        p_config.set(new_name, P_PATH, str(config_path.parent))
        with open(config_path, 'w') as config_file:
            p_config.write(config_file)
    else:
        logger.error(f"Error: The specified project configuration '{config_path}' does not exist.")


def update_project_section(config_path, name, set_name, new_value):
    logger.debug(f"update project {name}.ini config set {set_name}")
    p_config = ConfigParser()
    exist = p_config.read(config_path)
    if exist:
        p_config.set(name, set_name, new_value)
        with open(config_path, 'w') as config_file:
            p_config.write(config_file)
    else:
        logger.error(f"Error: The specified project configuration '{config_path}' does not exist.")


def rename_section(cp: ConfigParser, section_from, section_to):
    items = cp.items(section_from)
    cp.add_section(section_to)
    for item in items:
        cp.set(section_to, item[0], item[1])
    cp.remove_section(section_from)
