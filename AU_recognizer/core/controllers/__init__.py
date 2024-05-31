from abc import ABC, abstractmethod

from AU_recognizer.core.views import View


class Controller(ABC):
    @abstractmethod
    def bind(self, v: View):
        raise NotImplementedError
