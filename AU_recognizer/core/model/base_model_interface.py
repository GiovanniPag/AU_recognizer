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

    @abstractmethod
    def get_mesh_from_params(self, params):
        """Restituisce una mesh a partire da parametri"""
        pass

    @staticmethod
    @abstractmethod
    def au_difference(mesh_neutral, mesh_list, normalization_params, project):
        """Restituisce file con differenze di punti mesh"""
        pass

    @abstractmethod
    def emoca_tag(self, diff_files, threshold, au_names):
        """Genera file con vertici taggati"""
        pass

    @staticmethod
    @abstractmethod
    def get_ui_for_fit_data(master_widget) -> CustomFrame:
        """Restituisce la struttura widget della UI per acquisire i parametri specifici di fitting per il
        modello"""
        pass
