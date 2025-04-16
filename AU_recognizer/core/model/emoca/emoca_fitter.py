from AU_recognizer.core.projects.fitting_dialog import FitDialog


def emoca_fit(master, fit_data, images_to_fit, project_data):
    FitDialog(master=master, fit_data=fit_data, images_to_fit=images_to_fit, project_data=project_data).show()
