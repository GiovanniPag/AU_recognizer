import logging
import tkinter as tk


class TextRedirector:
    """A class to redirect stdout and stderr to a Tkinter Text widget."""

    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, message):
        """Write the message to the text widget and auto-scroll."""
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, f"{message} \n", (self.tag,))
        self.text_widget.configure(state="disabled")
        self.text_widget.yview(tk.END)  # Auto-scroll to the bottom

    def flush(self):
        """Flush method for compatibility with stdout/stderr."""
        pass


class TextLogger(logging.Handler):
    """Custom logging handler to redirect logs to Tkinter Text widget."""

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        """Write log messages to the Text widget."""
        msg = self.format(record) + "\n"
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg, ("log",))
        self.text_widget.configure(state="disabled")
        self.text_widget.yview(tk.END)  # Auto-scroll to the bottom
