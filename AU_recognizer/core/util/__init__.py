import configparser
import os
import subprocess
from pathlib import Path

from AU_recognizer.core.util import config as c
from AU_recognizer.core.util.config import logger, nect_config
from AU_recognizer.core.util.constants import CONFIG, GUIDE_FILE, LOG_FOLDER, P_NAME, P_PATH, \
    P_DONE, I18N_YES_BUTTON, I18N_NO_BUTTON, F_OUTPUT, F_INPUT, DESKTOP_LIST


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
    logger.debug("windowing system: " + c.windowing_system)
    if c.windowing_system == "x11":  # unix
        x11_func()
    if c.windowing_system == "win32":  # windows
        win32_func()
    if c.windowing_system == "aqua":  # macos
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
    if c.operating_system == "Windows":
        windows_func()
    elif c.operating_system == "Darwin":
        darwin_func()
    else:
        try:
            else_func()
        except Exception as e:
            logger.exception("call_by_ws exception in else_func", e)


def asset(asset_name):
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
