import tkinter
from typing import Callable


class ScalingTracker:
    window_widgets_dict = {}  # contains window objects as keys with list of widget callbacks as elements
    widget_scaling = 1  # user values which multiply to detected window scaling factor
    window_scaling = 1

    @classmethod
    def get_widget_scaling(cls) -> float:
        return cls.widget_scaling

    @classmethod
    def get_window_scaling(cls) -> float:
        return cls.window_scaling

    @classmethod
    def set_widget_scaling(cls, widget_scaling_factor: float):
        cls.widget_scaling = max(widget_scaling_factor, 0.4)
        cls.update_scaling_callbacks_all()

    @classmethod
    def set_window_scaling(cls, window_scaling_factor: float):
        cls.window_scaling = max(window_scaling_factor, 0.4)
        cls.update_scaling_callbacks_all()

    @classmethod
    def get_window_root_of_widget(cls, widget):
        current_widget = widget
        while isinstance(current_widget, tkinter.Tk) is False and \
                isinstance(current_widget, tkinter.Toplevel) is False:
            current_widget = current_widget.master
        return current_widget

    @classmethod
    def update_scaling_callbacks_all(cls):
        for window, callback_list in cls.window_widgets_dict.items():
            for set_scaling_callback in callback_list:
                set_scaling_callback(cls.widget_scaling,
                                     cls.window_scaling)

    @classmethod
    def update_scaling_callbacks_for_window(cls, window):
        for set_scaling_callback in cls.window_widgets_dict[window]:
            set_scaling_callback(cls.widget_scaling,
                                 cls.window_scaling)

    @classmethod
    def add_widget(cls, widget_callback: Callable, widget):
        window_root = cls.get_window_root_of_widget(widget)

        if window_root not in cls.window_widgets_dict:
            cls.window_widgets_dict[window_root] = [widget_callback]
        else:
            cls.window_widgets_dict[window_root].append(widget_callback)

    @classmethod
    def remove_widget(cls, widget_callback, widget):
        window_root = cls.get_window_root_of_widget(widget)
        # noinspection PyBroadException
        try:
            cls.window_widgets_dict[window_root].remove(widget_callback)
        except Exception:
            pass

    @classmethod
    def remove_window(cls, window):
        # noinspection PyBroadException
        try:
            del cls.window_widgets_dict[window]
        except Exception:
            pass

    @classmethod
    def add_window(cls, window_callback, window):
        if window not in cls.window_widgets_dict:
            cls.window_widgets_dict[window] = [window_callback]
        else:
            cls.window_widgets_dict[window].append(window_callback)
