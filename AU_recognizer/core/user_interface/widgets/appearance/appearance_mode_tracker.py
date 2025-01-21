from typing import Callable

from AU_recognizer.core.util import darkdetect


class AppearanceModeTracker:
    callback_list = []
    appearance_mode = 0  # Light (standard)

    @classmethod
    def init_appearance_mode(cls):
        new_appearance_mode = cls.detect_appearance_mode()

        if new_appearance_mode != cls.appearance_mode:
            cls.appearance_mode = new_appearance_mode
            cls.update_callbacks()

    @classmethod
    def add(cls, callback: Callable):
        cls.callback_list.append(callback)

    @classmethod
    def remove(cls, callback: Callable):
        try:
            cls.callback_list.remove(callback)
        except ValueError:
            return

    @staticmethod
    def detect_appearance_mode() -> int:
        try:
            if darkdetect.theme() == "Dark":
                return 1  # Dark
            else:
                return 0  # Light
        except NameError:
            return 0  # Light

    @classmethod
    def update_callbacks(cls):
        if cls.appearance_mode == 0:
            for callback in cls.callback_list:
                try:
                    callback("Light")
                except Exception:
                    continue

        elif cls.appearance_mode == 1:
            for callback in cls.callback_list:
                try:
                    callback("Dark")
                except Exception:
                    continue

    @classmethod
    def get_mode(cls) -> int:
        return cls.appearance_mode

    @classmethod
    def set_appearance_mode(cls, mode_string: str):
        new_appearance_mode = cls.appearance_mode
        if mode_string.lower() == "dark":
            new_appearance_mode = 1
        elif mode_string.lower() == "light":
            new_appearance_mode = 0

        if new_appearance_mode != cls.appearance_mode:
            cls.appearance_mode = new_appearance_mode
            cls.update_callbacks()
