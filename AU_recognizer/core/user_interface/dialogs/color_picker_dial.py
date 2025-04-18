import colorsys
import tkinter as tk
from typing import Callable

import numpy as np
from PIL import Image, ImageTk

from AU_recognizer.core.user_interface import CustomFrame, ThemeManager, CustomEntry, \
    CustomButton, CustomLabel, CustomToplevel


class AskColor(CustomToplevel):

    def __init__(self,
                 master: any = None,
                 title: str = "Choose Color",
                 initial_color: str = "#ffffff",
                 bg_color: str = None,
                 fg_color: str = None,
                 button_color: str = None,
                 button_hover_color: str = None,
                 text: str = "OK",
                 corner_radius: int = 24,
                 slider_border: int = 1,
                 command: Callable = None,
                 **button_kwargs):

        super().__init__(master=master)
        self.r_label = None
        self.r_entry = None
        self.g_label = None
        self.g_entry = None
        self.b_label = None
        self.b_entry = None
        self.switch_mode_button = None
        self._color = None
        self.title(title)
        self.resizable(width=False, height=False)
        # noinspection PyTypeChecker
        self.transient(self.master)
        self.after(100, self.lift)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.after(10)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.bg_color = self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["fg_color"]) if bg_color is None else bg_color
        self.fg_color = self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["top_fg_color"]) if fg_color is None else fg_color
        self.button_color = self._apply_appearance_mode(
            ThemeManager.theme["CustomButton"]["fg_color"]) if button_color is None else button_color
        self.button_hover_color = self._apply_appearance_mode(
            ThemeManager.theme["CustomButton"]["hover_color"]) if button_hover_color is None else button_hover_color
        self.button_text = text
        self.command = command
        self.corner_radius = corner_radius
        self.slider_border = min(10, slider_border)
        self.configure(bg_color=self.bg_color)
        self.frame = CustomFrame(master=self, fg_color=self.fg_color, bg_color=self.bg_color)
        self.frame.grid(padx=20, pady=20, sticky="nswe")

        self.mode = "RGB"
        self.hue, self.saturation, self.value = self.hex_to_hsv(initial_color)
        self.gradient_cache = {}

        # Gradient rectangle
        self.gradient_canvas = tk.Canvas(self.frame, width=300, height=200, bd=0, highlightthickness=0)
        self.gradient_canvas.pack()
        self.gradient_image_tk = ImageTk.PhotoImage(image=Image.new("RGB", (300, 200), color=(255, 0, 0)))
        self.gradient_canvas.create_image((0, 0), image=self.gradient_image_tk, anchor="nw")
        self.gradient_target = self.gradient_canvas.create_oval(-5, -5, 5, 5, outline="black", width=2)
        self.gradient_canvas.bind("<Button-1>", self.select_color)
        self.gradient_canvas.bind("<B1-Motion>", self.select_color)
        # Color preview
        self.preview_frame = CustomFrame(master=self.frame,
                                         fg_color=initial_color,
                                         bg_color=self.bg_color, width=40,
                                         height=40)
        self.preview_frame.pack(pady=5)
        # Hue slider
        self.hue_canvas = tk.Canvas(self.frame, height=15, width=300,
                                    highlightthickness=0,
                                    bg=self.fg_color)
        self.hue_canvas.pack(pady=10)
        self.hue_canvas.bind("<Button-1>", self.select_hue)
        self.hue_canvas.bind("<B1-Motion>", self.select_hue)
        self.create_hue_gradient()
        self.hue_target = self.hue_canvas.create_oval(-5, -5, 5, 5, outline="black", width=2)

        # Input fields
        self.input_frame = CustomFrame(master=self.frame, fg_color=self.fg_color, bg_color=self.bg_color)
        self.input_frame.pack(pady=10, fill="both")
        self.create_input_fields()
        self.button = CustomButton(master=self.frame, text=self.button_text, height=50,
                                   corner_radius=self.corner_radius,
                                   fg_color=self.button_color,
                                   hover_color=self.button_hover_color, command=self._ok_event, **button_kwargs)
        self.button.pack(fill="both", padx=10, pady=20)
        self.grab_set()
        self.set_color(hex=initial_color, update_gradient=True)

    def create_hue_gradient(self):
        """Create a hue gradient for the hue slider."""
        for i in range(300):
            hue = i / 300
            r, g, b = self.hsv_to_rgb(hue, 1, 1)
            color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            self.hue_canvas.create_line(i, 0, i, 15, fill=color)

    def update_gradient(self):
        """Update the gradient rectangle based on the selected hue."""
        width, height = 300, 200
        # Create saturation and value arrays (using broadcasting)
        x = np.arange(width)  # array of x coordinates (saturation)
        y = np.arange(height)  # array of y coordinates (value)

        saturation = x / width  # Saturation for each x
        value = 1 - y / height  # Value for each y
        # Now mesh the saturation and value grids
        s, v = np.meshgrid(saturation, value)
        # Vectorize the HSV to RGB conversion
        # Apply the vectorized function over the meshgrid
        r, g, b = self.hsv_to_rgb_vec(self.hue, s, v)
        # Scale RGB values to the range [0, 255] and fill the gradient array
        gradient_array = np.dstack((r, g, b)) * 255
        gradient_array = gradient_array.astype(np.uint8)
        # Create a PIL image from the NumPy array
        gradient_image = Image.fromarray(gradient_array)
        # Convert the PIL image to Tkinter-compatible format
        self.gradient_image_tk.paste(gradient_image)

    def set_color(self, update_gradient=False, **kwargs):
        """Set the color based on the input format (HSV, HSL, HEX, RGB)."""
        if 'hex' in kwargs:
            self.hue, self.saturation, self.value = self.hex_to_hsv(kwargs['hex'])
        elif 'rgb' in kwargs:
            self.hue, self.saturation, self.value = self.rgb_to_hsv(*kwargs['rgb'])
        elif 'hsv' in kwargs:
            self.hue, self.saturation, self.value = kwargs['hsv']
        elif 'hls' in kwargs:
            self.hue, self.saturation, self.value = self.hls_to_hsv(*kwargs['hls'])
        else:
            raise ValueError("Invalid color format. Please provide 'hex', 'rgb', 'hsv', or 'hsl'.")

        self.update_preview()
        # Update gradient for the new hue
        if update_gradient:
            self.update_gradient()
        # Update entries
        self.update_fields()
        # Update targets
        self.update_targets()
        if self.command:
            self.command(self.hsv_to_hex(self.hue, self.saturation, self.value))

    def update_preview(self):
        """Update the preview frame to show the current color."""
        self.preview_frame.configure(fg_color=self.hsv_to_hex(self.hue, self.saturation, self.value))

    def update_targets(self):
        """Update the position of the targets on the hue slider and gradient."""
        # Update hue target
        hue_x = int(self.hue * 300)
        self.hue_canvas.coords(self.hue_target, hue_x - 5, 2, hue_x + 5, 12)
        # Update gradient target
        gradient_x = int(self.saturation * 300)
        gradient_y = int((1 - self.value) * 200)
        self.gradient_canvas.coords(self.gradient_target, gradient_x - 5, gradient_y - 5, gradient_x + 5,
                                    gradient_y + 5)

    def create_input_fields(self):
        """Create RGB, HEX, or HLS input fields."""
        rgb_frame = CustomFrame(master=self.input_frame, fg_color=self.fg_color, bg_color=self.bg_color)
        rgb_frame.pack(pady=10, fill="both")
        self.r_label = CustomLabel(rgb_frame, text="R:")
        self.r_label.grid(row=0, column=0)
        self.r_entry = CustomEntry(rgb_frame, corner_radius=self.corner_radius)
        self.r_entry.grid(row=0, column=1)
        self.r_entry.bind("<FocusOut>", self.update_from_entries)
        self.r_entry.bind("<Return>", self.update_from_entries)

        self.g_label = CustomLabel(rgb_frame, text="G:")
        self.g_label.grid(row=0, column=2)
        self.g_entry = CustomEntry(rgb_frame, corner_radius=self.corner_radius)
        self.g_entry.grid(row=0, column=3)
        self.g_entry.bind("<FocusOut>", self.update_from_entries)
        self.g_entry.bind("<Return>", self.update_from_entries)

        self.b_label = CustomLabel(rgb_frame, text="B:")
        self.b_label.grid(row=0, column=4)
        self.b_entry = CustomEntry(rgb_frame, corner_radius=self.corner_radius)
        self.b_entry.grid(row=0, column=5)
        self.b_entry.bind("<FocusOut>", self.update_from_entries)
        self.b_entry.bind("<Return>", self.update_from_entries)

        self.switch_mode_button = CustomButton(self.input_frame, text="RGB ↕", command=self.switch_mode)
        self.switch_mode_button.pack(pady=10, fill="both")

    def update_from_entries(self, _=None):
        """Update the color based on the current input fields."""
        try:
            if self.mode == "RGB":
                # Validate and update color from RGB entries
                r = int(self.r_entry.get())
                g = int(self.g_entry.get())
                b = int(self.b_entry.get())
                r, g, b = r / 255, g / 255, b / 255
                if 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1:
                    self.set_color(rgb=(r, g, b), update_gradient=True)
                else:
                    raise ValueError
            elif self.mode == "HEX":
                # Validate and update color from HEX entry
                hex_color = self.r_entry.get().strip()
                if len(hex_color) == 7 and hex_color.startswith("#"):
                    self.set_color(hex=hex_color, update_gradient=True)
                else:
                    raise ValueError
            elif self.mode == "HLS":
                # Validate and update color from HSL entries
                h = float(self.r_entry.get())
                s = float(self.g_entry.get())
                light = float(self.b_entry.get())
                # Normalize HSL values to 0-1 range
                h = h / 360
                s = s / 100
                light = light / 100
                if 0 <= h <= 1 and 0 <= s <= 1 and 0 <= light <= 1:
                    self.set_color(hls=(h, light, s), update_gradient=True)
                else:
                    raise ValueError
        except ValueError:
            # Reset the fields if input is invalid
            self.update_fields()

    def switch_mode(self):
        """Switch between RGB, HEX, and HSL input modes."""
        if self.mode == "RGB":
            self.mode = "HEX"
            self.switch_mode_button.configure(text="HEX ↕")
            self.update_input_fields("HEX")
        elif self.mode == "HEX":
            self.mode = "HLS"
            self.switch_mode_button.configure(text="HSL ↕")
            self.update_input_fields("HLS")
        else:
            self.mode = "RGB"
            self.switch_mode_button.configure(text="RGB ↕")
            self.update_input_fields("RGB")

    def update_fields(self):
        if self.mode == "RGB":
            self.r_entry.delete(0, tk.END)
            self.g_entry.delete(0, tk.END)
            self.b_entry.delete(0, tk.END)
            # Populate RGB entries
            r, g, b = [int(c * 255) for c in self.hsv_to_rgb(self.hue, self.saturation, self.value)]
            self.r_entry.insert(0, r)
            self.g_entry.insert(0, g)
            self.b_entry.insert(0, b)
        elif self.mode == "HEX":
            self.r_entry.delete(0, tk.END)
            self.r_entry.insert(0, self.hsv_to_hex(self.hue, self.saturation, self.value))
        elif self.mode == "HLS":
            self.r_entry.delete(0, tk.END)
            self.g_entry.delete(0, tk.END)
            self.b_entry.delete(0, tk.END)
            h, light, s = self.hsv_to_hls(self.hue, self.saturation, self.value)
            h = int(h * 360)  # Hue in degrees
            s = int(s * 100)  # Saturation in percentage
            light = int(light * 100)  # Lightness in percentage
            self.r_entry.insert(0, h)
            self.g_entry.insert(0, s)
            self.b_entry.insert(0, light)

    def update_input_fields(self, mode):
        """Update the input fields to match the selected mode."""
        if mode == "RGB":
            self.r_label.configure(text="R:")
            self.g_label.configure(text="G:")
            self.b_label.configure(text="B:")
            self.r_entry.delete(0, tk.END)
            self.g_entry.delete(0, tk.END)
            self.b_entry.delete(0, tk.END)
            # Populate RGB entries
            r, g, b = [int(c * 255) for c in self.hsv_to_rgb(self.hue, self.saturation, self.value)]
            self.r_entry.insert(0, r)
            self.g_entry.insert(0, g)
            self.b_entry.insert(0, b)
        elif mode == "HEX":
            self.r_label.configure(text="HEX:")
            self.g_label.grid_remove()
            self.b_label.grid_remove()
            self.g_entry.grid_remove()
            self.b_entry.grid_remove()
            self.r_entry.delete(0, tk.END)
            self.r_entry.insert(0, self.hsv_to_hex(self.hue, self.saturation, self.value))
        elif mode == "HLS":
            self.r_label.configure(text="H:")
            self.g_label.configure(text="S:")
            self.b_label.configure(text="L:")
            self.g_label.grid()
            self.b_label.grid()
            self.g_entry.grid()
            self.b_entry.grid()
            self.r_entry.delete(0, tk.END)
            self.g_entry.delete(0, tk.END)
            self.b_entry.delete(0, tk.END)
            h, light, s = self.hsv_to_hls(self.hue, self.saturation, self.value)
            h = int(h * 360)  # Hue in degrees
            s = int(s * 100)  # Saturation in percentage
            light = int(light * 100)  # Lightness in percentage
            self.r_entry.insert(0, h)
            self.g_entry.insert(0, s)
            self.b_entry.insert(0, light)

    def get(self):
        self._color = self.hsv_to_hex(self.hue, self.saturation, self.value)
        self.master.wait_window(self)
        return self._color

    def _ok_event(self, _=None):
        self._color = self.hsv_to_hex(self.hue, self.saturation, self.value)
        self.grab_release()
        self.destroy()

    def _on_closing(self):
        self._color = self.hsv_to_hex(self.hue, self.saturation, self.value)
        self.grab_release()
        self.destroy()

    def select_hue(self, event):
        """Select a hue based on the position on the hue slider."""
        x = max(0, min(300, event.x))
        self.hue = x / 300
        self.update_gradient()
        self.set_color(hsv=(self.hue, self.saturation, self.value), update_gradient=True)

    def select_color(self, event):
        """Select a color from the gradient rectangle."""
        x = max(0, min(300, event.x))
        y = max(0, min(200, event.y))
        self.saturation = x / 300
        self.value = 1 - y / 200
        self.set_color(hsv=(self.hue, self.saturation, self.value))

    @staticmethod
    def hls_to_hsv(h, light, s):
        # Convert HSL to RGB
        r, g, b = colorsys.hls_to_rgb(h, light, s)
        return colorsys.rgb_to_hsv(r, g, b)

    @staticmethod
    def hsv_to_hls(h, s, v):
        # Convert HSL to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return colorsys.rgb_to_hls(r, g, b)

    @staticmethod
    def rgb_to_hsv(r, g, b):  # rgb normalized to [0,1]
        return colorsys.rgb_to_hsv(r, g, b)

    # Color conversion helpers
    @staticmethod
    def hsv_to_rgb(h, s, v):
        return colorsys.hsv_to_rgb(h, s, v)

    @staticmethod
    def hex_to_hsv(hex_color):
        # Remove the "#" if it exists
        hex_color = hex_color.lstrip('#')
        # Convert hex to RGB
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # Return HSV values
        return h, s, v

    @staticmethod
    def hsv_to_hex(h, s, v):
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        # Convert RGB to Hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        return hex_color

    @staticmethod
    def hsv_to_rgb_vec(h, s, v):
        """Vectorized version of HSV to RGB conversion."""
        h = np.asarray(h)
        s = np.asarray(s)
        v = np.asarray(v)

        i = np.floor(h * 6) % 6
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        i = i.astype(int)

        r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
        g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
        b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])

        return r, g, b
