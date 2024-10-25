import json

from AU_recognizer.core.util.config import write_config, nect_config, logger
from AU_recognizer.core.util.constants import *


class I18N:
    # Internationalization
    def __init__(self):
        self.language = nect_config[CONFIG][LANGUAGE]
        self.i18n_path = Path(nect_config[CONFIG][I18N_PATH])
        logger.debug(f"application language: {self.language} i18n path: {self.i18n_path}")
        self.__resource_language()

    def change_language(self, language) -> bool:
        if self.language == language:
            logger.debug("tried to change language but it's the same language")
            return False
        else:
            logger.debug(f"try to change language from: {self.language} to: {language}")
            nect_config[CONFIG][LANGUAGE] = language
            write_config()
            self.language = language
            self.__resource_language()
            return True

    def __resource_language(self):
        read_file = None
        try:
            logger.debug(f"try to open the language json: {self.i18n_path}/{self.language}.json")
            read_file = open(self.i18n_path / (self.language+".json"), "r")
        except IOError:
            logger.exception("file not found")
            logger.debug(f"open config language json file: {self.i18n_path}/en.json")
            read_file = open(self.i18n_path / "en.json", "r")
        finally:
            with read_file:
                logger.debug("save json data in variables")
                data = json.load(read_file)
                # program title
                self.title = data['title']
                # menu cascade items
                self.menu_file = data['menu']['menu_file']
                self.menu_help = data['menu']['menu_help']
                # menu file items
                self.menu_file_new = data['menu']['menu_file']["new_project"]
                self.menu_file_open = data['menu']['menu_file']["open_project"]
                self.menu_file_settings = data['menu']['menu_file']["settings"]
                self.menu_file_exit = data['menu']['menu_file']["exit"]
                # menu help items
                self.menu_help_language = data['menu']['menu_help']["language"]
                self.menu_help_language_it = data['menu']['menu_help']["language"]['it']
                self.menu_help_language_en = data['menu']['menu_help']["language"]['en']
                self.menu_help_about = data['menu']['menu_help']["about"]
                self.menu_help_logs = data['menu']['menu_help']["logs"]
                self.menu_help_guide = data['menu']['menu_help']["guide"]
                # tree menu context
                self.menut_add_images = data['menu_contextual']['tree_view']["add_images"]
                self.menut_selected_project = data['menu_contextual']['tree_view']["selected_project"]
                self.menut_select_project = data['menu_contextual']['tree_view']["select_project"]
                self.menut_close_project = data['menu_contextual']['tree_view']["close_project"]
                self.menut_delete_project = data['menu_contextual']['tree_view']["delete_project"]
                self.menut_selected_file = data['menu_contextual']['tree_view']["selected_file"]
                self.menut_open_file = data['menu_contextual']['tree_view']["open_file"]
                self.menut_delete_file = data['menu_contextual']['tree_view']["delete_file"]
                self.menut_rename_file = data['menu_contextual']['tree_view']["rename_file"]
                # dialog buttons
                self.dialog_buttons = data['dialog']['buttons']
                # project exist error dialog
                self.project_message = data['dialog']["project_message"]
                # about dialog
                self.p_options_dialog = data['dialog']["p_options"]
                # rename path dialog
                self.p_rename_dialog = data['dialog']["p_rename"]
                # choose folder dialog
                self.choose_folder_dialog = data['dialog']["choose_folder_pr"]
                # choose folder dialog general
                self.choose_folder_dialog_g = data['dialog']["choose_folder"]
                # choose images dialog
                self.choose_images_dialog = data['dialog']["choose_images"]
                # confirmation dialog
                self.confirmation_dialog = data['dialog']["confirmation_dialog"]
                # tree view
                self.tree_view = data['tree_view']
                # selected file view
                self.selected_file_view = data['selected_file']
                # selected project view
                self.selected_project_view = data['selected_project']
                # project actions
                self.project_actions_fit = data['project_actions']['fit']
                self.project_actions_au_rec = data['project_actions']['au_rec']
                # entry buttons
                self.entry_buttons = data['entry_buttons']
                # radio buttons
                self.radio_buttons = data['radio_buttons']
                # settings dialog
                self.settings_dialog = data['dialog']["setting_dialog"]
                # select image dialog
                self.im_sel_dialog = data['dialog']["image_selection"]
                # open gl viewer
                self.gl_viewer = data['gl_viewer']
                # tooltips
                self.tooltips = data['tooltips']


i18n = I18N()
