import tkinter as tk
from pathlib import Path
from tkinter import ttk, HORIZONTAL, VERTICAL

from AU_recognizer.core.controllers import Controller
from AU_recognizer.core.util import call_by_ws, config as c, check_if_folder_exist, check_if_is_project
from AU_recognizer.core.util.config import logger, nect_config, purge_option_config
from AU_recognizer.core.util.constants import OPEN_PROJECTS, P_PATH
from AU_recognizer.core.util.language_resource import i18n
from AU_recognizer.core.controllers.controller import MenuController, TreeController, \
    Viewer3DController, SelectedFileController, SelectedProjectController, ProjectActionController, \
    TreeViewMenuController
from AU_recognizer.core.views.view import MenuBar, View, ProjectTreeView, Viewer3DView, ScrollWrapperView, \
    SelectedFileView, ProjectInfoView, ProjectActionView, TreeViewMenu


# main window of AU_rec MVC app
class AURecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.option_add('*tearOff', tk.FALSE)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.title(i18n.title)
        c.windowing_system = self.tk.call('tk', 'windowingsystem')
        logger.debug("windowing system: " + c.windowing_system)

        self.devices = []
        self.open_projects = {}
        self.selected_project = None

        self.__recover_model()
        self.__create_style()
        self.__create_gui()
        self.__create_controllers()
        self.__bind_controllers()

        logger.debug("make app full-screen")
        call_by_ws(x11_func=lambda: self.attributes('-zoomed', True), aqua_func=lambda: self.state("zoomed"),
                   win32_func=lambda: self.state("zoomed"))

    @staticmethod
    def __bind_controller(controller: Controller, view: View):
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

    @staticmethod
    def __create_style():
        logger.debug("create style")
        # Initialize style
        s = ttk.Style()
        # Create style used by default for all Frames
        s.configure('TFrame', background='green')
        # Create style for the first frame
        s.configure('Right.TFrame', background='red')
        s.configure('Left.TFrame', background='purple')

        s.configure('Image.TFrame', background='blue')
        s.configure('Viewer3DView.TFrame', background='yellow')

    # create all view classes
    def __create_gui(self):
        logger.debug("create gui")

        # create menu bar
        self.menu_bar = MenuBar(self)
        # create mainframe
        logger.debug("main frame initialization")
        mainframe = ttk.PanedWindow(self, orient=HORIZONTAL)
        mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        logger.debug("left frame initialization")
        left_frame = ttk.PanedWindow(self, orient=VERTICAL, style="Left.TFrame")
        logger.debug("right frame initialization")
        right_frame = ttk.PanedWindow(self, orient=VERTICAL, style="Right.TFrame")
        logger.debug("center frame initialization")
        center_frame = ttk.PanedWindow(self, orient=VERTICAL)
        # create views
        # left views
        scroll_wrapper = ScrollWrapperView(master=left_frame)
        self.project_tree_view = ProjectTreeView(master=scroll_wrapper)
        scroll_wrapper.add(self.project_tree_view)
        self.selected_file = SelectedFileView(self)
        left_frame.add(scroll_wrapper, weight=3)
        left_frame.add(self.selected_file, weight=2)

        # center views
        self.project_info = ProjectInfoView(self)
        self.project_actions = ProjectActionView(self)
        center_frame.add(self.project_info, weight=1)
        center_frame.add(self.project_actions, weight=5)

        # right views
        self.viewer = Viewer3DView(self, style="Viewer3DView.TFrame")
        right_frame.add(self.viewer, weight=5)

        mainframe.add(left_frame, weight=1)
        mainframe.add(center_frame, weight=5)
        mainframe.add(right_frame, weight=10)
        # create context menu
        self.tree_menu = TreeViewMenu(self.project_tree_view)

    def __create_controllers(self):
        logger.debug("create controllers")
        # create controllers
        self.menu_controller = MenuController(self)
        self.tree_menu_controller = TreeViewMenuController(self)
        self.tree_controller = TreeController(self.tree_menu_controller, self)
        self.viewer_controller = Viewer3DController(self)
        self.selected_controller = SelectedFileController(self)
        self.info_controller = SelectedProjectController(self)
        self.action_controller = ProjectActionController(self)

    def __bind_controllers(self):
        logger.debug("bind all controllers")
        # bind views to controllers
        self.__bind_controller(controller=self.menu_controller, view=self.menu_bar)
        self.__bind_controller(controller=self.tree_menu_controller, view=self.tree_menu)
        self.__bind_menu(controller=self.tree_controller, menu=self.tree_menu_controller)
        self.__bind_controller(controller=self.tree_controller, view=self.project_tree_view)
        self.__bind_controller(controller=self.viewer_controller, view=self.viewer)
        self.__bind_controller(controller=self.selected_controller, view=self.selected_file)
        self.__bind_controller(controller=self.info_controller, view=self.project_info)
        self.__bind_controller(controller=self.action_controller, view=self.project_actions)

        # attach menu_bar
        logger.debug("attach menu_bar")
        self['menu'] = self.menu_bar
        # attach virtual event controllers
        logger.debug("attach virtual event controllers for <<LanguageChange>>")
        self.bind("<<LanguageChange>>", lambda event: self.update_app_language())
        logger.debug("attach virtual event controllers for <<UpdateTree>>")
        self.bind("<<UpdateTree>>", lambda event: self.tree_controller.update_tree_view(populate_root=True, sort_tree=True))
        logger.debug("attach virtual event controllers for <<selected_project>>")
        self.bind("<<selected_project>>", self.select_project)
        logger.debug("attach virtual event controllers for <<selected_file>>")
        self.bind("<<selected_item>>", self.select_file)
        logger.debug("attach virtual event controllers for <<open_file>>")
        self.bind("<<open_file>>", self.open_file)

    def select_project(self, event):
        project = self.tree_controller.get_last_selected_project()
        path = None
        if project:
            path = project[P_PATH]
        try:
            self.selected_project = self.open_projects[str(path)] if path else None
            logger.debug(f"select project event {event} path {path}, selected project {self.selected_project}")
            self.info_controller.update_view(self.selected_project)
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
        self.selected_controller.update_view(file)
        self.viewer_controller.update_view(file)

    def open_file(self, event):
        logger.debug(f"open file event {event}")
        self.selected_controller.open_file()

    def update_app_language(self):
        logger.debug("update app language")
        self.title(i18n.title)
        self.menu_bar.update_language()
        self.tree_menu.update_language()
        self.project_tree_view.update_language()
        self.viewer.update_language()
        self.selected_file.update_language()
        self.project_info.update_language()
        self.project_actions.update_language()

    def can_grab_focus(self):
        return not self.tree_menu.is_open