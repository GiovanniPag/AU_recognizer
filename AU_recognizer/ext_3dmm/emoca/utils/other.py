import sys
from pathlib import Path


def class_from_str(string, module=None, none_on_fail=False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, string):
        cl = getattr(module, string)
        return cl
    elif string.lower() == 'none' or none_on_fail:
        return None  # type: ignore
    raise RuntimeError(f"Class '{string}' not found.")


def get_path_to_externals() -> Path:
    import AU_recognizer.external
    return Path(AU_recognizer.external.__file__)
