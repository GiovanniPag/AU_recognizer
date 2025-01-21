from pathlib import Path

# Desktops
DESKTOP_LIST = ["Desktop", "Scrivania"]
# config file path
CONFIG_FILE = Path.cwd() / "AU_recognizer" / "var" / ".conf" / "AU_rec.ini"
# config file sections
OPEN_PROJECTS = "open projects"
CONFIG = "config"
VIEWER = "viewer_3d"
# config file config section items
LANGUAGE = "language"
I18N_PATH = "i18n_path"
LOGGER = "logger"
LOGGER_PATH = "logger_path"
LOG_FOLDER = "log_folder"
PROJECTS_FOLDER = "projects_folder"
MODEL_FOLDER = "model_folder"
GUIDE_FILE = "guide_file"
# config file config section default values
LANGUAGE_DEFAULT = "it"
I18N_PATH_DEFAULT = "AU_recognizer/var/i18n/"
LOGGER_DEFAULT = "AULogger"
LOGGER_PATH_DEFAULT = "AU_recognizer/var/.conf/logging.ini"
LOG_FOLDER_DEFAULT = "AU_recognizer/var/.logs/"
PROJECTS_FOLDER_DEFAULT = "AU_recognizer/projects/"
MODEL_FOLDER_DEFAULT = "AU_recognizer/../emoca/assets/EMOCA/models"
GUIDE_FILE_DEFAULT = "AU_recognizer/var/guide.pdf"
# config file viewer section items
FILL_COLOR = "fill_color"
LINE_COLOR = "line_color"
CANVAS_COLOR = "canvas_color"
POINT_COLOR = "point_color"
POINT_SIZE = "point_size"
MOVING_STEP = "moving_step"
SKY_COLOR = "sky_color"
GROUND_COLOR = "ground_color"

# config file viewer section default values
FILL_COLOR_DEFAULT = "#000000"
LINE_COLOR_DEFAULT = "#0000FF"
CANVAS_COLOR_DEFAULT = "#FFFFFF"
POINT_COLOR_DEFAULT = "#131313"
SKY_COLOR_DEFAULT = "#87CEEB"
GROUND_COLOR_DEFAULT = "#604020"
POINT_SIZE_DEFAULT = 1
MOVING_STEP_DEFAULT = 10

# project config file items
P_NAME = "name"
P_PATH = "path"
P_INDEX = "index"

P_EMPTY = ""
P_DONE = "done"
P_TRUE = "true"
P_FALSE = "false"
# project folders
F_INPUT = "input"
F_OUTPUT = "output"

# tk icons
BASENAME_ICON = "::tk::icons::"
INFORMATION_ICON = "information"
WARNING_ICON = "warning"
ERROR_ICON = "error"
FULL_QUESTION_ICON = "::tk::icons::question"

# # i18n constants
I18N_TITLE = "title"
I18N_NAME = "name"
I18N_MESSAGE = "message"
I18N_DETAIL = "detail"
I18N_NAME_TIP = "name_tip"
I18N_BACK_BUTTON = "back"
I18N_CONFIRM_BUTTON = "confirm"
I18N_YES_BUTTON = "yes"
I18N_NO_BUTTON = "no"
I18N_CANCEL_BUTTON = "cancel"
I18N_SAVE_BUTTON = "save"
I18N_CLOSE_BUTTON = "close"
I18N_FIT_SEL_BUTTON = "fit_sel"
I18N_FIT_ALL_BUTTON = "fit_all"

# menu options
M_UNDERLINE = "underline"
M_MASTER = "master"
M_LABEL = "label"
M_INDEX = "index"
M_STATE = "state"
M_DEFAULT_STATE = "default_state"
M_STATE_NORMAL = "normal"
M_ACCELERATOR = "accelerator"
M_VALUE = "value"
M_VARIABLE = "variable"
M_RADIO = "radio"
M_COMMAND = "command"

# menu commands names
M_FILE = "file"
M_EDIT = "edit"
M_HELP = "help"
M_NEW = "new"
M_OPEN = "open"
M_SETTINGS = "settings"
M_EXIT = "exit"
M_UNDO = "undo"
M_REDO = "redo"
M_CUT = "cut"
M_COPY = "copy"
M_PASTE = "paste"
M_DELETE = "delete"
M_LANGUAGE = "language"
M_ABOUT = "about"
M_LOGS = "logs"
M_GUIDE = "guide"
M_IT = "it"
M_EN = "en"

# menu tree view context command names
MT_SEL_P = "selected_project"
MT_ADD_IMAGES = "add_img_to_project"
MT_SELECT_P = "select_project"
MT_CLOSE_P = "close_project"
MT_DELETE_P = "delete_project"
MT_SEL_F = "selected_file"
MT_OPEN_F = "open_file"
MT_DELETE_F = "delete_file"
MT_RENAME_F = "rename_file"

# tree view strings
T_COLUMNS = "columns"
T_CENTER = "center"
T_NAME = "#0"
T_NAME_HEADING = "name"
T_SIZE = "size"
T_MODIFIED = "modified"
T_END = "end"
T_TEXT = "text"
T_VALUES = "values"
T_FILE_TYPE = "type"
T_DATA = "data"
# tree view menu
TM_FILE = "file"
TM_PROJECT = "project"
TM_PROJECT_PATH = "project_path"
TM_FILE_PATH = "file_path"
# tree view dialog
TD_FITTED = "fitted"
TD_IMAGES = "images"
TD_HIDE_FITTED = "hide_fitted"

# project view strings
PV_PATH = "path"
PV_NAME = "name"
PV_DONE = "done"
PV_NOT_DONE = "not_done"
PV_NO = "no_project"

# file view strings
FV_FILE = "file"
FV_SUFFIX = "suffix"
FV_OPEN_S = "open_s"
FV_TYPE = "type"
FV_NO = "no_file"
FV_U = "unknown"
FV_F = "folder"

# project actions strings
PA_NAME = "name"
PA_DISABLED = "disabled"
PA_NORMAL = "normal"

# Fit View Radiobutton
RADIO_TEXT = "r_text"
RADIO_BTN = "r_btn"
R_COARSE = "coarse"
R_DETAIL = "detail"
MF_SELECT_IMG = "select_img"
MF_MODEL = "model"
MF_SAVE_IMAGES = "save_images"
MF_SAVE_CODES = "save_codes"
MF_SAVE_MESH = "save_mesh"
MF_FIT_MODE = "fit_mode"

# Setting Dialog
GENERAL_TAB = "general_tab"
VIEWER_TAB = "viewer_3d_tab"

# Define arrow symbols
up_arrow = '\u2191'
down_arrow = '\u2193'
left_arrow = '\u2190'
right_arrow = '\u2192'

# viewer 3d GL strings
GL_U_POINTER = "pointer"
GL_U_TYPE = "type"
GL_U_VALUE = "value"
GL_SOLID = "solid"
GL_WIREFRAME = "wireframe"
GL_C_POINTS = "points"
GL_DEFAULT = "default"
GL_V_COLOR = "v_color"
GL_C_TEXTURE = "texture"
GL_NO = "no"
GL_NORMAL = "normal"
GL_C_NORMAL_MAP = "normal_map"
