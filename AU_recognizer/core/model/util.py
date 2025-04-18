from AU_recognizer.core.user_interface.dialogs.fitting_dialog import FitDialog


def image_fit(master, fit_data, images_to_fit, project_data):
    FitDialog(master=master, fit_data=fit_data, images_to_fit=images_to_fit, project_data=project_data).show()
