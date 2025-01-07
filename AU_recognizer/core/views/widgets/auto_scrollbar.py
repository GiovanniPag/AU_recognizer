import tkinter as tk
from tkinter import ttk as ttk

from AU_recognizer.core.util import logger
from AU_recognizer.core.views import GeometryManager


class AutoScrollbar(ttk.Scrollbar):
    def __init__(self, master, geometry: GeometryManager = GeometryManager.GRID, column_grid=1, row_grid=0,
                 **kwargs):
        super().__init__(master, **kwargs)
        self.geometry = geometry
        self.column = column_grid
        self.row = row_grid

    """Create a scrollbar that hides itself if it's not needed."""

    def set(self, lo, hi):
        logger.debug(f"hide scrollbar if not needed")
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            if self.geometry is GeometryManager.GRID:
                self.grid_forget()
            else:
                self.pack_forget()
        else:
            if str(self.cget("orient")) == tk.HORIZONTAL:
                if self.geometry is GeometryManager.GRID:
                    self.grid(column=self.column, row=self.row, sticky=(tk.W, tk.E))
                else:
                    self.pack(fill=tk.X, side=tk.BOTTOM)
            else:
                if self.geometry is GeometryManager.GRID:
                    self.grid(column=self.column, row=self.row, sticky=(tk.N, tk.S))
                else:
                    self.pack(fill=tk.Y, side=tk.RIGHT)
        ttk.Scrollbar.set(self, lo, hi)

    def place(self, **kw):
        raise tk.TclError("cannot use place with this widget")
