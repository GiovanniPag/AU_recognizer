import platform
import tkinter as tk

from AU_recognizer.core.views import AutoScrollbar, GeometryManager


class ScrollFrame(tk.Frame):
    def __init__(self, master, debug=False):
        super().__init__(master)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.viewPort = tk.Frame(self.canvas)
        if debug:
            self.viewPort.configure(background="#bbaaee")
        self.vsb = AutoScrollbar(self, geometry=GeometryManager.GRID, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.viewPort, anchor="nw", tags="self.viewPort")

        self.viewPort.bind("<Configure>", self.on_frame_configure)
        # bind an event whenever the size of the viewPort frame changes.
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        # bind an event whenever the size of the canvas frame changes.
        self.viewPort.bind('<Enter>', self.on_enter)
        # bind wheel events when the cursor enters the control
        self.viewPort.bind('<Leave>', self.on_leave)
        # unbind wheel events when the cursor leaves the control
        self.on_frame_configure(None)
        # perform an initial stretch on render, otherwise the scroll region has a tiny border until the first resize

    def on_frame_configure(self, _):
        """ Reset the scroll region to encompass the inner frame """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # whenever the size of the frame changes, alter the scroll region respectively.

    def on_canvas_configure(self, event):
        """ Reset the canvas window to encompass inner frame when required """
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        # whenever the size of the canvas changes alter the window region respectively.

    def on_mouse_wheel(self, event):  # cross platform scroll wheel event
        if platform.system() == 'Windows':
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif platform.system() == 'Darwin':
            self.canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

    def on_enter(self, event):
        # bind wheel events when the cursor enters the control
        if self.vsb.winfo_ismapped():
            if platform.system() == 'Linux':
                self.canvas.bind_all("<Button-4>", self.on_mouse_wheel)
                self.canvas.bind_all("<Button-5>", self.on_mouse_wheel)
            else:
                self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)

    def on_leave(self, event):
        # unbind wheel events when the cursor leaves the control
        if platform.system() == 'Linux':
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        else:
            self.canvas.unbind_all("<MouseWheel>")
