import configparser
import os
import re
import subprocess
from configparser import ConfigParser
from pathlib import Path
from typing import Union

from . import logger, nect_config, windowing_system, operating_system, write_config
from .constants import CONFIG, GUIDE_FILE, LOG_FOLDER, P_NAME, P_PATH, F_OUTPUT, F_INPUT, DESKTOP_LIST, OPEN_PROJECTS


def open_guide():
    logger.debug(" open " + GUIDE_FILE + " file")
    path = nect_config[CONFIG][GUIDE_FILE]
    open_path_by_os(path)


def open_log_folder():
    logger.debug("Open logs folder")
    path = Path(nect_config[CONFIG][LOG_FOLDER])
    open_path_by_os(path)


def open_path_by_os(path: Path):
    call_by_os(windows_func=lambda: os.startfile(path), darwin_func=lambda: subprocess.Popen(["open", path]),
               else_func=lambda: subprocess.Popen(["xdg-open", path]))


def call_by_ws(x11_func, win32_func, aqua_func):
    if not (callable(x11_func) and callable(win32_func) and callable(aqua_func)):
        logger.exception("one of the argument function is not callable")
        return
    logger.debug("windowing system: " + windowing_system)
    if windowing_system == "x11":  # unix
        x11_func()
    if windowing_system == "win32":  # windows
        win32_func()
    if windowing_system == "aqua":  # macos
        aqua_func()


def check_if_folder_exist(path: Path, folder_name="") -> bool:
    logger.debug(f"check if folder {str(path / folder_name)} exist")
    if folder_name:
        return (path / folder_name).is_dir()
    else:
        return path.is_dir()


def check_if_file_exist(path: Path, file_name) -> bool:
    logger.debug(f"check if file {str(path / file_name)} exist")
    return (path / file_name).is_file()


def validate(project_metadata: configparser.ConfigParser, project_name, project_path: Path):
    logger.debug(f"check if project {project_name} is valid")
    valid = bool(project_metadata)
    if valid:
        valid = project_metadata.has_section(project_name) and project_metadata.has_option(project_name, P_NAME) and \
                project_metadata[project_name][P_NAME] == project_name and project_metadata.has_option(project_name,
                                                                                                       P_PATH) and Path(
            project_metadata[project_name][P_PATH]).samefile(project_path) and check_if_folder_exist(
            project_path / F_OUTPUT) and check_if_folder_exist(project_path / F_INPUT)
    return valid


def check_if_is_project(path: Path, project_name):
    logger.debug(f"check if project {project_name} exist and is valid")
    project_conf = []
    is_project = check_if_file_exist(path, project_name + ".ini")
    if is_project:
        project_conf = configparser.ConfigParser()
        project_conf.read(path / (project_name + ".ini"))
        is_project = validate(project_metadata=project_conf, project_name=project_name, project_path=path)
    return is_project, project_conf


def call_by_os(windows_func, darwin_func, else_func):
    if not (callable(windows_func) and callable(darwin_func) and callable(else_func)):
        logger.exception("one of the argument function is not callable")
        return
    if operating_system == "Windows":
        windows_func()
    elif operating_system == "Darwin":
        darwin_func()
    else:
        try:
            else_func()
        except Exception as e:
            logger.exception("call_by_ws exception in else_func", e)


def asset(asset_name: Union[str, Path]) -> Path:
    return Path(__file__).parent / ".." / ".." / "var" / "asset" / asset_name


def get_desktop_path():
    # Get the user's home directory
    home = Path.home()

    # Determine the desktop folder based on common desktop folder names
    desktop_folders = DESKTOP_LIST

    for desktop in desktop_folders:
        desktop_path = home / desktop
        if desktop_path.is_dir():
            return desktop_path
    # If no common desktop folder is found, fall back to the default "Desktop" folder
    return home


def retrieve_files_from_path(path, file_type):
    return path.glob(file_type)


def hex_to_float_rgba(hex_c: str, alpha: bool = False):
    h = hex_c.lstrip("#")
    if alpha:
        rgba = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4, 6)]
    else:
        rgba = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1]
    return rgba


def hex_to_float_rgb(hex_c: str):
    h = hex_c.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return rgb


def pop_from_dict_by_set(dictionary: dict, valid_keys: set) -> dict:
    """ remove and create new dict with key value pairs of dictionary, where key is in valid_keys """
    new_dictionary = {}

    for key in list(dictionary.keys()):
        if key in valid_keys:
            new_dictionary[key] = dictionary.pop(key)

    return new_dictionary


def check_kwargs_empty(kwargs_dict, raise_error=False) -> bool:
    """ returns True if kwargs are empty, False otherwise, raises error if not empty """
    if len(kwargs_dict) > 0:
        if raise_error:
            raise ValueError(f"{list(kwargs_dict.keys())} are not supported arguments. Look at the documentation for "
                             f"supported arguments.")
        else:
            return True
    else:
        return False


def rename_path(path_to_rename,
                new_name):
    path = Path(path_to_rename)
    new_name = new_name + path.suffix if path.is_file() else new_name
    path = path.rename(Path(path.parent, new_name))
    logger.info(f"File '{path}' successfully renamed.")
    return path


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


def check_name(new_val):
    logger.debug(f"validate project name {new_val}")
    return re.match('^[a-zA-Z0-9-_]*$', new_val) is not None and len(new_val) <= 50


def check_file_name(new_val):
    logger.debug(f"validate file name {new_val}")
    return re.match(r'^[a-zA-Z0-9-_\\.]*$', new_val) is not None


def check_num(new_val):
    logger.warning(f"validate only int {new_val}")
    return re.match('^[0-9]*$', new_val) is not None and len(new_val) <= 3


def sizeof_fmt(num, suffix="B"):
    logger.debug(f"format {num} in bytes")
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"

