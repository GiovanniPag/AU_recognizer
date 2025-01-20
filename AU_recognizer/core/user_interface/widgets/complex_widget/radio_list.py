import tkinter as tk

from AU_recognizer.core.user_interface import CustomLabel, CustomRadioButton
from AU_recognizer.core.user_interface.views import View
from AU_recognizer.core.util import i18n, RADIO_TEXT, RADIO_BTN


class RadioList(View):
    def __init__(self, master=None, list_title="no_text", default="", data=None, orientation=tk.HORIZONTAL, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.list_title = list_title
        self.radio_var = tk.StringVar(value=default)
        self.label_radio_group = CustomLabel(master=self, text=i18n.entry_buttons[self.list_title])
        self.radio_buttons = {}
        self.orientation = orientation
        for btn_name in data:
            text = tk.StringVar(value=i18n.radio_buttons[btn_name])
            self.radio_buttons[btn_name] = {
                RADIO_TEXT: text,
                RADIO_BTN: CustomRadioButton(master=self, text=btn_name, textvariable=text, variable=self.radio_var,
                                             value=btn_name)
            }
        self.update_language()

    def create_view(self):
        self.label_radio_group.pack(side=tk.LEFT, expand=True, fill=tk.X)
        for btn_name, btn_data in self.radio_buttons.items():
            btn = btn_data[RADIO_BTN]
            if self.orientation == tk.HORIZONTAL:
                btn.pack(side=tk.LEFT, padx=5, pady=5)
            elif self.orientation == tk.VERTICAL:
                btn.pack(anchor=tk.W, padx=5, pady=5)

    def get_value(self):
        return self.radio_var.get()

    def update_language(self):
        self.label_radio_group.configure(text=i18n.entry_buttons[self.list_title])
        for btn_name, btn_data in self.radio_buttons.items():
            btn_data[RADIO_BTN].update_language()
