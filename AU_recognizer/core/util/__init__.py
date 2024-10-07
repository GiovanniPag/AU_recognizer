import configparser
import os
import subprocess
import time
from pathlib import Path
import re
from pprint import pprint

import numpy as np

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


def time_me(f):
    """Decorator function to time functions' runtime in ms"""

    def wrapper(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        logger.debug(f'function: {f.__name__} took {(time.time() - start) * 1000:.4f}ms')
        return res

    return wrapper


def extract_data(file_path):
    """
    @brief: Extracts vertices, faces, texture coordinates, and material information from an .obj file.

    @param file_path: Path of a .obj file

    @ret  : vertices (a numpy array of shape (n, 3) where n is the number of vertices)
    @ret  : faces (a list of tuples, each containing face indices, texture coordinate indices, and material name)
    @ret  : texcoords (a numpy array of texture coordinates)
    @ret  : materials (a dictionary with material properties)
    """
    vertices = []
    colors = []  # Add this to store vertex colors
    faces = []
    texcoords = []  # List to store texture coordinates
    materials = {}
    current_material = None

    append_vertex = vertices.append
    append_color = colors.append
    append_texcoord = texcoords.append
    with file_path.open() as file:
        for line in file:
            if line.startswith("v "):
                coordinates = list(map(float, line[2:].split()))
                append_vertex(coordinates)
            elif line.startswith("vt "):
                texcoord = list(map(float, line[3:].split()))
                append_texcoord(texcoord)
            elif line.startswith("mtllib "):
                mtl_files = line.split()[1:]
                for mtl_file in mtl_files:
                    materials.update(load_materials(
                        Path(mtl_file) if Path(mtl_file).is_absolute() else (file_path.parent / mtl_file).resolve()))
            elif line.startswith("usemtl "):
                current_material = line.split()[1]
            elif line.startswith("f "):
                face_data = line[2:].split()
                face_indices = []
                texcoord_indices = []
                for vertex in face_data:
                    parts = vertex.split('/')
                    face_indices.append(int(parts[0]) - 1)
                    texcoord_indices.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1)
                faces.append((face_indices, texcoord_indices, current_material))

    return np.array(vertices), np.array(colors), faces, np.array(texcoords), materials


def load_materials(mtl_file_path):
    materials = {}
    current_material = None

    with open(mtl_file_path) as mtl_file:
        for line in mtl_file:
            if line.startswith('newmtl '):
                current_material = line.split()[1]
                materials[current_material] = {}
            elif line.startswith('map_Kd ') and current_material:
                f = line.split()[1]
                materials[current_material]['texture'] = Path(f) if Path(f).is_absolute() else (
                        mtl_file_path.parent / f).resolve()
            elif line.startswith('disp ') and current_material:
                f = line.split()[1]
                materials[current_material]['normal'] = Path(f) if Path(f).is_absolute() else (
                        mtl_file_path.parent / f).resolve()

    return materials


def prepare_data_for_opengl(vertices, colors, faces, textcoords, materials):
    batched_data = {}

    for face in faces:
        v_idx, vt_idx, material = face
        if material not in batched_data:
            batched_data[material] = {'vertex_data': [], 'index_data': []}

        vertex_data = batched_data[material]['vertex_data']
        index_data = batched_data[material]['index_data']

        for i in range(len(v_idx)):
            vi = v_idx[i]
            vti = vt_idx[i]
            verts = []
            verts.extend(vertices[vi])
            verts.extend(textcoords[vti])
            vertex_data.append(verts)
            index_data.append(len(vertex_data) - 1)

    # Convert vertex and index data to numpy arrays
    for material in batched_data:
        batched_data[material]['vertex_data'] = np.array(batched_data[material]['vertex_data'],
                                                         dtype=np.float32)
        batched_data[material]['index_data'] = np.array(batched_data[material]['index_data'], dtype=np.uint32)
    return batched_data, materials


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
