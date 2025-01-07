import tkinter as tk
from math import floor


class DiscreteStep(tk.Scale):
    def __init__(self, master=None, step=1, **kw):
        super().__init__(master, **kw)
        self.step = floor(step)
        self.variable: tk.IntVar = kw.get("variable")
        self.value_list = list(range(int(kw.get("from_")), int(kw.get("to") + 1), self.step))
        self.configure(command=self.value_check)

    def value_check(self, value):
        new_value = min(self.value_list, key=lambda x: abs(x - float(value)))
        self.variable.set(value=new_value)
