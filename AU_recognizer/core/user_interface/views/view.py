from abc import abstractmethod

from ..widgets.core_widget_classes import CustomFrame


class View(CustomFrame):

    @abstractmethod
    def create_view(self):
        raise NotImplementedError

    @abstractmethod
    def update_language(self):
        raise NotImplementedError


