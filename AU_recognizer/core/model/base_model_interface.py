from abc import ABC, abstractmethod

from AU_recognizer.core.user_interface import CustomFrame


class BaseModelInterface(ABC):
    @staticmethod
    @abstractmethod
    def get_models_list():
        pass

    @staticmethod
    @abstractmethod
    def fit(fit_data, images_to_fit, additional_data):
        pass

    @staticmethod
    @abstractmethod
    def au_difference(mesh_neutral, mesh_list, normalization_params, project):
        """Restituisce file con differenze di punti mesh"""
        pass

    @staticmethod
    @abstractmethod
    def emoca_tag(diff_to_tag, threshold, project_data):
        """Genera file con vertici taggati"""
        pass

    @staticmethod
    @abstractmethod
    def get_ui_for_fit_data(master_widget) -> CustomFrame:
        """Restituisce la struttura widget della UI per acquisire i parametri specifici di fitting per il
        modello"""
        pass
