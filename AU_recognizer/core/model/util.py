import re

from AU_recognizer.core.user_interface.dialogs.fitting_dialog import FitDialog


def image_fit(master, fit_data, images_to_fit, project_data):
    FitDialog(master=master, fit_data=fit_data, images_to_fit=images_to_fit, project_data=project_data).show()


def extract_codes(filename):
    match = re.search(r"(\d+(?:_\d+)*)", filename)  # Find numbers separated by '_'
    return match.group(1) if match else ""  # Return the matched codes or empty string
