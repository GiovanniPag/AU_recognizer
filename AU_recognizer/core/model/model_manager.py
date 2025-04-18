import inspect
import pkgutil
import importlib

from AU_recognizer.core.model.base_model_interface import BaseModelInterface


def list_models():
    models = []
    package = 'AU_recognizer.core.model.models'

    for _, name, _ in pkgutil.iter_modules([package.replace('.', '/')]):
        module = importlib.import_module(f'{package}.{name}')
        for item in dir(module):
            obj = getattr(module, item)
            if (inspect.isclass(obj) and
                    issubclass(obj, BaseModelInterface) and
                    obj is not BaseModelInterface):
                models.append(obj)
    return models


def load_model_class(name: str):
    models = list_models()
    for model in models:
        if any(name in p.name for p in model.get_models_list()):
            return model
