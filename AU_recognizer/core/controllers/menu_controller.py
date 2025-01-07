from pathlib import Path
from tkinter import filedialog

from .base_controller import Controller
from ..util import logger, nect_config, CONFIG, LANGUAGE, M_ABOUT, M_COMMAND, M_LOGS, open_log_folder, \
    M_GUIDE, open_guide, M_IT, M_EN, M_NEW, M_OPEN, M_SETTINGS, M_EXIT, check_if_is_project, ERROR_ICON, i18n, \
    I18N_BACK_BUTTON, check_if_folder_exist, PROJECTS_FOLDER, I18N_TITLE, add_to_open_projects, create_project_folder, \
    store_open_project
from ..views.dialogs.dialog import DialogProjectOptions
from ..views.dialogs.dialog_util import open_message_dialog
from AU_recognizer.core.views.views.view import MenuBar


class MenuController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.master = master
        self.view = None

    def bind(self, v: MenuBar):
        logger.debug("bind in menu controller")
        self.view = v
        self.view.create_view()
        self.view.language.set(nect_config[CONFIG][LANGUAGE])
        # bind command to menu items
        logger.debug("menu controller bind commands")
        # menu help
        self.view.update_command_or_cascade(M_ABOUT, {M_COMMAND: lambda: open_message_dialog(self.master, "about")})
        self.view.update_command_or_cascade(M_LOGS, {M_COMMAND: lambda: open_log_folder()})
        self.view.update_command_or_cascade(M_GUIDE, {M_COMMAND: lambda: open_guide()})
        # menu help language
        self.view.update_command_or_cascade(M_IT, {M_COMMAND: lambda: self.language_change(M_IT)})
        self.view.update_command_or_cascade(M_EN, {M_COMMAND: lambda: self.language_change(M_EN)})
        # menu file
        self.view.update_command_or_cascade(M_NEW, {M_COMMAND: lambda: self.create_new_project()})
        self.view.update_command_or_cascade(M_OPEN, {M_COMMAND: lambda: self.open_project()})
        self.view.update_command_or_cascade(M_SETTINGS, {M_COMMAND: lambda: self.open_settings_dialog()})
        self.view.update_command_or_cascade(M_EXIT, {M_COMMAND: lambda: self.close_app()})

    def open_project(self):
        logger.debug("open project")
        opening = True
        while opening:
            path = self.choose_folder()
            if path != "":
                logger.debug("chosen project path: " + str(path))
                is_project, metadata = check_if_is_project(path, path.name)
                if is_project:
                    if str(path) in self.master.open_projects:
                        logger.debug("project is already open")
                        open_message_dialog(self.master, "project_already_open", ERROR_ICON)
                    else:
                        logger.debug("project opened successfully")
                        self.master.add_project(metadata, path.name)
                        add_to_open_projects(path)
                        open_message_dialog(self.master, "project_open_success")
                        logger.debug("menu_bar <<UpdateTree>> event generation")
                        self.master.event_generate("<<UpdateTree>>")
                        opening = False
                else:
                    logger.debug("selected folder is not a project")
                    open_message_dialog(self.master, "not_project", ERROR_ICON)
            else:
                opening = False

    def create_new_project(self):
        logger.debug("create new project")
        creating = True
        while creating:
            path = self.choose_folder()
            if path != "":
                logger.debug("chosen project path: " + str(path))
                name = self.ask_project_name()
                if name != i18n.dialog_buttons[I18N_BACK_BUTTON]:
                    if name != "":
                        logger.debug("chosen project name: " + name)
                        exist = check_if_folder_exist(path, name)
                        if not exist:
                            create_project_folder(path / name)
                            metadata = store_open_project(path, name)
                            self.master.add_project(metadata, name)
                            logger.debug("menu_bar <<UpdateTree>> event generation")
                            self.master.event_generate("<<UpdateTree>>")
                            logger.debug("project created successfully")
                        else:
                            logger.debug("invalid name")
                            open_message_dialog(self.master, "project_exist", ERROR_ICON)
                    creating = False
                else:
                    logger.debug("go back to choose path")
            else:
                creating = False

    def close_app(self):
        logger.debug("close app")
        self.master.destroy()

    def ask_project_name(self):
        logger.debug("open p_options dialog, ask project name")
        p_name = DialogProjectOptions(master=self.master, has_back=True).show()
        return p_name

    def choose_folder(self):
        logger.debug("choose directory")
        dir_name = filedialog.askdirectory(initialdir=nect_config[CONFIG][PROJECTS_FOLDER], mustexist=True,
                                           parent=self.master, title=i18n.choose_folder_dialog[I18N_TITLE])
        if dir_name != () and dir_name:
            dir_name = Path(dir_name)
            logger.debug("chosen path: " + str(dir_name))
            return dir_name
        else:
            logger.debug("abort choosing")
            return ""

    def language_change(self, language):
        changed = i18n.change_language(language)
        if changed:
            logger.debug("menu_bar <<LanguageChange>> event generation")
            self.master.event_generate("<<LanguageChange>>")

    def open_settings_dialog(self, page=""):
        logger.debug("open settings dialog")
        self.master.open_settings(page)
