from pathlib import Path
from tkinter import ttk, HORIZONTAL, FALSE, NSEW, Menu
from typing import Union

from AU_recognizer.core.util import (i18n, logger, config, call_by_ws, nect_config, OPEN_PROJECTS,
                                     check_if_folder_exist, check_if_is_project, purge_option_config, P_PATH)
from AU_recognizer.core.user_interface import *
from AU_recognizer.core.user_interface.views import (View, MenuBar, ProjectTreeView,
                                                     ProjectActionView, Viewer3DView, TreeViewMenu)
from AU_recognizer.core.user_interface.dialogs.complex_dialog import SettingsDialog
from AU_recognizer.core.controllers import Controller, MenuController, TreeViewMenuController, TreeController, \
    Viewer3DController, ProjectActionController

set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


# main window of AU_rec MVC app
class AURecognizer(CustomTk):
    def __init__(self):
        super().__init__()
        self.option_add('*tearOff', FALSE)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.title(i18n.title)
        config.windowing_system = self.tk.call('tk', 'windowingsystem')
        logger.debug("windowing system: " + config.windowing_system)
        self.devices = []
        self.open_projects = {}
        self.selected_project = None
        self.__recover_model()
        self.__create_gui()
        self.__create_controllers()
        self.__bind_controllers()
        logger.debug("make app full-screen")
        call_by_ws(x11_func=lambda: self.attributes('-zoomed', True), aqua_func=lambda: self.state("zoomed"),
                   win32_func=lambda: self.state("zoomed"))

    @staticmethod
    def __bind_controller(controller: Controller, view: Union[View, Menu]):
        logger.debug(f"bind {controller.__class__} to {view.__class__}")
        controller.bind(view)

    @staticmethod
    def __bind_menu(controller: Controller, menu: Controller):
        logger.debug(f"bind {controller.__class__} to {menu.__class__}")
        menu.controller = controller

    def __recover_model(self):
        logger.debug("recover open projects")
        # recover open projects from file
        open_projects = nect_config[OPEN_PROJECTS]
        for option in open_projects:
            logger.debug("opening project: " + open_projects[option] + ", from path: " + option)
            path = Path(option)
            purge = not check_if_folder_exist(path)
            if not purge:
                logger.debug("project: " + open_projects[option] + " folder exist")
                is_project, metadata = check_if_is_project(path, open_projects[option])
                if is_project:
                    logger.debug("project: " + option + " exists, add it to open projects")
                    self.add_project(metadata, open_projects[option])
                else:
                    purge = True
            if purge:  # purge not found
                purge_option_config(option)

    def open_settings(self, page=""):
        SettingsDialog(master=self, page=page).show()

    def add_project(self, project_data, project_name):
        logger.debug(f"add {project_data} from {project_data[project_name][P_PATH]} to open projects")
        self.open_projects[project_data[project_name][P_PATH]] = project_data

    def remove_project(self, project_path):
        logger.debug(f"remove project {project_path} from open project")
        del self.open_projects[project_path]
        purge_option_config(project_path)

    def rename_project(self, old_path, new_name, new_metadata):
        self.remove_project(old_path)
        self.add_project(new_metadata, new_name)

    # create all view classes
    def __create_gui(self):
        logger.debug("create gui")
        # create menu bar
        self.menu_bar = MenuBar(self)
        # create mainframe
        logger.debug("main frame initialization")
        mainframe = ttk.PanedWindow(self, orient=HORIZONTAL)
        mainframe.grid(column=0, row=0, sticky=NSEW)
        logger.debug("left frame initialization")
        self.project_tree_view = ProjectTreeView(master=self)
        mainframe.add(self.project_tree_view, weight=2)
        logger.debug("center frame initialization")
        self.project_actions = ProjectActionView(self)
        mainframe.add(self.project_actions, weight=3)
        logger.debug("right frame initialization")
        self.viewer = Viewer3DView(self)
        mainframe.add(self.viewer, weight=15)
        # create context menu
        self.tree_menu = TreeViewMenu(self.project_tree_view)

    def __create_controllers(self):
        logger.debug("create controllers")
        # create controllers
        self.menu_controller = MenuController(self)
        self.tree_menu_controller = TreeViewMenuController(self)
        self.tree_controller = TreeController(self.tree_menu_controller, self)
        self.viewer_controller = Viewer3DController(self)
        self.action_controller = ProjectActionController(self)

    def __bind_controllers(self):
        logger.debug("bind all controllers")
        # bind views to controllers
        self.__bind_controller(controller=self.menu_controller, view=self.menu_bar)
        self.__bind_controller(controller=self.tree_menu_controller, view=self.tree_menu)
        self.__bind_menu(controller=self.tree_controller, menu=self.tree_menu_controller)
        self.__bind_controller(controller=self.tree_controller, view=self.project_tree_view)
        self.__bind_controller(controller=self.viewer_controller, view=self.viewer)
        self.__bind_controller(controller=self.action_controller, view=self.project_actions)

        # attach menu_bar
        logger.debug("attach menu_bar")
        self.configure(menu=self.menu_bar)
        # attach virtual event controllers
        logger.debug("attach virtual event controllers for <<LanguageChange>>")
        self.bind("<<LanguageChange>>", lambda event: self.update_app_language())
        logger.debug("attach virtual event controllers for <<ViewerChange>>")
        self.bind("<<ViewerChange>>", lambda event: self.update_3d_viewer())
        logger.debug("attach virtual event controllers for <<UpdateTree>>")
        self.bind("<<UpdateTree>>",
                  lambda event: self.tree_controller.update_tree_view(populate_root=True, sort_tree=True))
        logger.debug("attach virtual event controllers for <<UpdateTreeSmall>>")
        self.bind("<<UpdateTreeSmall>>",
                  lambda event: self.tree_controller.update_tree_view(populate_root=False, sort_tree=True))
        logger.debug("attach virtual event controllers for <<selected_project>>")
        self.bind("<<selected_project>>", self.select_project)
        logger.debug("attach virtual event controllers for <<selected_file>>")
        self.bind("<<selected_item>>", self.select_file)

    def select_project(self, event):
        project = self.tree_controller.get_last_selected_project()
        path = None
        if project:
            path = project[P_PATH]
        try:
            self.selected_project = self.open_projects[str(path)] if path else None
            logger.debug(f"select project event {event} path {path}, selected project {self.selected_project}")
            self.action_controller.select_project(self.selected_project)
        except KeyError as e:
            logger.error(f"Errore: {e}, file doesn't exist, update tree view")
            self.event_generate("<<UpdateTree>>")

    def select_file(self, event):
        file = self.tree_controller.get_last_selected_file()
        logger.debug(f"select file event {event} data {file}")
        if file is not None and not Path(file[P_PATH]).exists():
            logger.error(f"Errore: {file}, file doesn't exist, update tree view")
            self.event_generate("<<UpdateTree>>")
            file = None
        self.viewer_controller.update_view(file)

    def update_app_language(self):
        logger.debug("update app language")
        self.title(i18n.title)
        self.menu_bar.update_language()
        self.tree_menu.update_language()
        self.project_tree_view.update_language()
        self.viewer.update_language()
        self.project_actions.update_language()

    def can_grab_focus(self):
        return not self.tree_menu.is_open

    def update_3d_viewer(self):
        self.viewer.update_3d()
