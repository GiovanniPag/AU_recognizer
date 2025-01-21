from typing import Callable

from AU_recognizer.core.user_interface import CustomFrame, CustomButton, CustomEntry


class CustomSpinbox(CustomFrame):
    def __init__(self, *args,
                 width: int = 100,
                 height: int = 32,
                 step_size: float = 1,
                 default: float = 0,
                 min_value: float = 0,
                 max_value: float = 0,
                 use_float: bool = True,
                 command: Callable = None,
                 **kwargs):
        self.min_value = float(min_value) if use_float else int(min_value)
        self.max_value = float(max_value) if use_float else int(max_value)
        self.use_float = use_float
        super().__init__(*args, width=width, height=height, **kwargs)

        self.step_size = float(step_size) if use_float else int(step_size)
        self.command = command

        self.configure(fg_color=("gray78", "gray28"))

        self.grid_columnconfigure((0, 2), weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.subtract_button = CustomButton(self, text="-", width=height - 6, height=height - 6,
                                            command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.entry = CustomEntry(self, width=width - 70, height=height - 6, border_width=0)
        self.entry.grid(row=0, column=1, columnspan=1, padx=3, pady=3, sticky="nsew")

        self.add_button = CustomButton(self, text="+", width=height - 6, height=height - 6,
                                       command=self.add_button_callback)
        self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

        # default value
        self.entry.insert(0, f"{float(default) if use_float else int(default)}")
        # All elements on mousewheel event
        self.entry.bind("<MouseWheel>", self.on_mouse_wheel)
        self.subtract_button.bind("<MouseWheel>", self.on_mouse_wheel)
        self.add_button.bind("<MouseWheel>", self.on_mouse_wheel)
        self.bind("<MouseWheel>", self.on_mouse_wheel)

    def add_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            value = float(self.entry.get()) + self.step_size if self.use_float else int(self.entry.get()) + int(
                self.step_size)
            if value <= self.max_value:
                self.entry.delete(0, "end")
                self.entry.insert(0, f"{value if self.use_float else int(value)}")
        except ValueError:
            return

    def subtract_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            value = float(self.entry.get()) - self.step_size if self.use_float \
                else int(self.entry.get()) - int(self.step_size)
            if value >= self.min_value:
                self.entry.delete(0, "end")
                self.entry.insert(0, f"{value if self.use_float else int(value)}")
        except ValueError:
            return

    def get(self) -> int:
        try:
            return float(self.entry.get()) if self.use_float else int(self.entry.get())
        except ValueError:
            return 0.0 if self.use_float else 0

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.add_button_callback()
        else:
            self.subtract_button_callback()

    def set(self, value: int):
        self.entry.delete(0, "end")
        self.entry.insert(0, f"{float(value) if self.use_float else int(value)}")
