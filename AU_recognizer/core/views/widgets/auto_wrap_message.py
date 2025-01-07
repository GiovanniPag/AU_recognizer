import tkinter as tk


class AutoWrapMessage(tk.Message):
    def __init__(self, master, margin=8, **kwargs):
        super().__init__(master, **kwargs)
        self.margin = margin
        self.bind("<Configure>", lambda event: event.widget.configure(width=event.width - self.margin))
