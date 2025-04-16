import shutil
from pathlib import Path

from AU_recognizer.core.user_interface.dialogs.dialog import DialogMessage, DialogAsk, DialogPathRename
from AU_recognizer.core.util import (INFORMATION_ICON, logger, i18n, I18N_TITLE, I18N_MESSAGE, I18N_DETAIL,
                                     WARNING_ICON, I18N_NO_BUTTON, I18N_YES_BUTTON, check_if_folder_exist,
                                     check_if_file_exist, rename_path, ERROR_ICON)


def open_message_dialog(master, message, icon=INFORMATION_ICON):
    logger.debug(f"open project_{message} dialog")
    data = i18n.project_message[message]
    DialogMessage(master=master,
                  title=data[I18N_TITLE],
                  message=data[I18N_MESSAGE],
                  detail=data[I18N_DETAIL],
                  icon=icon).show()


def open_confirmation_dialogue(master, message, icon=WARNING_ICON):
    logger.debug(f"open confirmation_{message} dialogue")
    data = i18n.confirmation_dialog[message]
    return DialogAsk(master=master,
                     title=data[I18N_TITLE],
                     message=data[I18N_MESSAGE],
                     detail=data[I18N_DETAIL],
                     icon=icon).show()


def confirm_deletion(master, dialog_type, ask_confirmation=True):
    answer = I18N_NO_BUTTON
    if ask_confirmation:
        answer = open_confirmation_dialogue(master, dialog_type)
    return not ask_confirmation or (ask_confirmation and answer == I18N_YES_BUTTON)


def delete_path(path_to_delete, ask_confirmation=True, dialog_folder="delete_folder", dialog_file="delete_file",
                master=None):
    logger.debug(
        f"delete following path and all it's content: {path_to_delete}")
    path = Path(path_to_delete)
    removed = False
    if path.exists():
        # Remove the path (file or folder) and its contents if it's a folder
        try:
            if path.is_dir() and confirm_deletion(master, dialog_folder, ask_confirmation):
                shutil.rmtree(path)
                removed = True
                logger.info(f"Folder '{path}' and its contents successfully removed.")
            elif path.is_file() and confirm_deletion(master, dialog_file, ask_confirmation):
                path.unlink()
                removed = True
                logger.info(f"File '{path}' successfully removed.")
        except FileNotFoundError as e:
            print(f"Error while deleting {path}: {e.filename} not found.")
        except PermissionError as e:
            print(f"Error while deleting {path}: Permission denied - {e.filename}.")
        except OSError as e:
            logger.exception(f"Error while deleting {path}: {e.filename} - {e.strerror}")
    else:
        logger.error(f"Error: The specified path '{path}' does not exist.")
    return removed


def confirm_rename_path(path_to_rename,
                        master=None):
    logger.debug(
        f"rename following path: {path_to_rename}")
    path = Path(path_to_rename)
    renamed = False
    if path.exists():
        new_file_name = DialogPathRename(master=master).show()
        if new_file_name != "":
            logger.debug("chosen file name: " + new_file_name)
            exist = check_if_folder_exist(path.parent, new_file_name) or check_if_file_exist(path=path.parent,
                                                                                             file_name=new_file_name)
            if not exist:
                path = rename_path(path_to_rename=path_to_rename, new_name=new_file_name)
                renamed = True
            else:
                logger.debug("invalid name")
                open_message_dialog(master, "name_taken", ERROR_ICON)
    else:
        logger.error(f"Error: The specified path '{path}' does not exist.")
    return renamed, path if renamed else ""
