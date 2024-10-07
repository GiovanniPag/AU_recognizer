import math
import platform

import numpy as np

import tkinter as tk
from tkinter import ttk
from typing import Optional

from AU_recognizer import VIEWER, CANVAS_COLOR, MOVING_STEP, POINT_COLOR, POINT_SIZE, FILL_COLOR, LINE_COLOR, i18n, \
    up_arrow, left_arrow, down_arrow, right_arrow
from AU_recognizer.core.util import nect_config, time_me, extract_data
from AU_recognizer.core.util.geometry_3d import Geometry3D, axis_angle_to_quaternion, quaternion_multiply, \
    quaternion_to_euler, is_face_visible_2D
from AU_recognizer.core.views import View, ScaleLabel, CheckLabel


class Viewer3D(View):
    def __init__(self, placeholder, obj_file_path=None):
        super().__init__(master=placeholder)
        self._fill_color = str(nect_config[VIEWER][FILL_COLOR])
        self._line_color = str(nect_config[VIEWER][LINE_COLOR])
        self._canvas_color = tk.StringVar(value=str(nect_config[VIEWER][CANVAS_COLOR]))
        self._moving_step = int(nect_config[VIEWER][MOVING_STEP])
        self._point_size = int(nect_config[VIEWER][POINT_SIZE])
        self._point_color = str(nect_config[VIEWER][POINT_COLOR])
        # Initial canvas ratios
        self._canvas_w = 0
        self._canvas_h = 0
        self._geometry_handler = Geometry3D(self._canvas_w, self._canvas_h)
        # declare view items
        self._canvas: Optional[tk.Canvas] = None
        self.slider_frame = ttk.Frame(self)
        self._zoom_slider: Optional[ScaleLabel] = ScaleLabel(master=self.slider_frame, label_text="zoom_slider",
                                                             from_=10.0,
                                                             to=0.1, initial_value=self._geometry_handler.zoom,
                                                             orient=tk.HORIZONTAL,
                                                             command=self.__set_zoom)
        self._x_slider: Optional[ScaleLabel] = ScaleLabel(master=self.slider_frame, label_text="x_slider",
                                                          from_=-math.pi,
                                                          to=math.pi, initial_value=0,
                                                          orient=tk.HORIZONTAL,
                                                          command=self.__set_rotations)
        self._y_slider: Optional[ScaleLabel] = ScaleLabel(master=self.slider_frame, label_text="y_slider",
                                                          from_=-math.pi,
                                                          to=math.pi, initial_value=0,
                                                          orient=tk.HORIZONTAL,
                                                          command=self.__set_rotations)
        self._z_slider: Optional[ScaleLabel] = ScaleLabel(master=self.slider_frame, label_text="z_slider",
                                                          from_=-math.pi,
                                                          to=math.pi, initial_value=0,
                                                          orient=tk.HORIZONTAL,
                                                          command=self.__set_rotations)

        self._reset_rot: Optional[ttk.Button] = None
        self._no_fill = CheckLabel(master=self.slider_frame, label_text="no_fill", default=False,
                                   command=self.__changed)
        self.create_view()
        self.bind_events()
        self.__reset_rotation()

        # A flag used to only redraw the object when a change occurred
        self._changed = True
        self._is_updating = False
        if obj_file_path:
            self.__set_file(obj_file_path)

    def create_view(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # canvas
        self._canvas = tk.Canvas(self, bg=self._canvas_color.get())
        self._canvas.grid(row=0, column=0, columnspan=10, padx=10, pady=10, sticky=tk.NSEW)
        # Create a frame for sliders
        self.slider_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self._zoom_slider.create_view()
        self._zoom_slider.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self._x_slider.create_view()
        self._x_slider.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self._y_slider.create_view()
        self._y_slider.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self._z_slider.create_view()
        self._z_slider.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        self._reset_rot = ttk.Button(self.slider_frame, text=i18n.entry_buttons["reset_rot"],
                                     command=self.__reset_rotation)
        self._reset_rot.grid(row=0, column=4, padx=5, pady=5, sticky="e")

        ttk.Button(self.slider_frame, text=up_arrow, command=self.__move_up).grid(row=1, column=1, padx=5, pady=5,
                                                                                  sticky="e")
        ttk.Button(self.slider_frame, text=down_arrow, command=self.__move_down).grid(row=2, column=1, padx=5, pady=5,
                                                                                      sticky="e")
        ttk.Button(self.slider_frame, text=left_arrow, command=self.__move_left).grid(row=2, column=0, padx=5, pady=5,
                                                                                      sticky="e")
        ttk.Button(self.slider_frame, text=right_arrow, command=self.__move_right).grid(row=2, column=2, padx=5, pady=5,
                                                                                        sticky="e")
        self._no_fill.create_view()
        self._no_fill.grid(row=1, column=2, padx=5, pady=5, sticky="e")

    def bind_events(self):
        # Catch the canvas resize event
        self._canvas.bind("<Configure>", self.__resized)
        # Bind the scrolling on canvas event to zoom slider (Button-4/5 on linux)
        self._canvas.bind('<Button-4>', self.on_mouse_wheel)
        self._canvas.bind('<Button-5>', self.on_mouse_wheel)
        # zoom for Windows and MacOS, but not Linux
        self._canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        # Bind keys for movement
        self._canvas.bind_all('<KeyPress-w>', self.__move_up)
        self._canvas.bind_all('<KeyPress-s>', self.__move_down)
        self._canvas.bind_all('<KeyPress-a>', self.__move_left)
        self._canvas.bind_all('<KeyPress-d>', self.__move_right)
        self._canvas.bind_all('<KeyPress-Up>', self.__move_up)
        self._canvas.bind_all('<KeyPress-Down>', self.__move_down)
        self._canvas.bind_all('<KeyPress-Left>', self.__move_left)
        self._canvas.bind_all('<KeyPress-Right>', self.__move_right)

        # Bind mouse events for rotation
        self._canvas.bind("<Button-1>", self.__start_rotate)
        self._canvas.bind("<B1-Motion>", self.__rotate)

    def on_mouse_wheel(self, event):
        """Cross-platform mouse wheel scroll event for zooming"""
        if platform.system() == 'Windows':
            delta = event.delta / 120
        elif platform.system() == 'Darwin':
            delta = event.delta
        else:  # Assume Linux or other
            if event.num == 5:
                delta = 1
            elif event.num == 4:
                delta = -1
            else:
                delta = 0
        # Adjust the zoom slider based on the scroll direction
        if delta != 0:
            self._zoom_slider.set(self._zoom_slider.get() + delta * 0.1)

    def __get_canvas_shape(self):
        """returns the shape of the canvas holding the visualized frame"""
        self.update()
        return self._canvas.winfo_width(), self._canvas.winfo_height()

    def __resized(self, *args):
        """Callback to the window resize events"""
        w, h = self.__get_canvas_shape()
        if self._canvas_w != w or self._canvas_h != h:
            # Keep the object in the middle of the canvas
            self._geometry_handler.update_position((w - self._canvas_w) // 2, (h - self._canvas_h) // 2)
            self._canvas_w = w
            self._canvas_h = h
            self._geometry_handler.update_scale(width=self._canvas_w, height=self._canvas_h)
            self.__changed()

    def __changed(self, *args):
        """Signal to the rendering function that something has changed in the object"""
        self._changed = True

    def __reset_rotation(self):
        self._geometry_handler.reset_rotation()
        self._x_slider.set(0)
        self._y_slider.set(0)
        self._z_slider.set(0)
        self.__changed()

    def __move_up(self, event=None):
        self._geometry_handler.update_position(0, -1 * self._moving_step)
        self.__changed()

    def __move_down(self, event=None):
        self._geometry_handler.update_position(0, self._moving_step)
        self.__changed()

    def __move_left(self, event=None):
        self._geometry_handler.update_position(-1 * self._moving_step, 0)
        self.__changed()

    def __move_right(self, event=None):
        self._geometry_handler.update_position(self._moving_step, 0)
        self.__changed()

    def render(self):
        """Render the object on the screen"""
        if self._changed:
            # Delete all the previous points and lines in order to draw new ones
            self._canvas.delete("all")
            self.__update_colors()
            self.__draw_object()
            self._changed = False

    def __set_zoom(self, *args):
        self._geometry_handler.set_zoom(self._zoom_slider.get())
        self.__changed()

    def __set_rotations(self, *args):
        """Set the required rotations for the geometry handler"""
        x_angle = self._x_slider.get()
        y_angle = self._y_slider.get()
        z_angle = self._z_slider.get()
        self._geometry_handler.set_rotation_from_xyz(x_angle, y_angle, z_angle)
        self.__changed()

    def __change_fill_color(self, no_fill: bool = False):
        """Change the face fill color"""
        self._fill_color = "" if no_fill else str(nect_config[VIEWER][FILL_COLOR])

    def __draw_point(self, point: 'tuple(int, int)') -> None:
        """Draw a point on the canvas"""
        self._canvas.create_oval(point[0], point[1],
                                 point[0], point[1],
                                 width=self._point_size,
                                 fill=self._point_color)
        # for point in to_draw:
        #	if(point[0] < 0 or
        #	   point[1] < 0 or
        #	   point[0] > self.CANVAS_WIDTH or
        #	   point[1] > self.CANVAS_HEIGHT
        #	):
        #		continue # Don't draw points that are out of the screen
        #	# This is the slowest part of the GUI
        #	self.__draw_point(point)

    @time_me
    def __draw_faces(self, projected_points: dict) -> None:
        for face_idx, face in enumerate(self._geometry_handler.faces):
            # Grab the points that make up that specific face
            to_draw = [projected_points[vertex] for vertex in face]

            # Check if face is visible (not back-facing)
            # if is_face_visible_2D(to_draw):
            self._canvas.create_polygon(to_draw, outline=self._line_color, fill=self._fill_color)

    def __draw_object(self):
        """Draw the object on the canvas"""
        projected_points = self._geometry_handler.transform_object()
        self.__draw_faces(projected_points)

    def __update_colors(self):
        self.__change_fill_color(self._no_fill.get_value())

    def update_display(self):
        if not self._is_updating:
            self._is_updating = True
            self.render()
            self.after(10, self._reset_update_display)

    def _reset_update_display(self):
        self._is_updating = False
        self.update_display()

    def update_language(self):
        self._zoom_slider.update_language()
        self._x_slider.update_language()
        self._y_slider.update_language()
        self._z_slider.update_language()
        self._reset_rot.config(text=i18n.entry_buttons["reset_rot"])
        self._no_fill.update_language()

    def __start_rotate(self, event):
        """Start capturing the initial mouse click position for rotation"""
        self._start_orientation = self._geometry_handler.orientation
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

    def __rotate(self, event):
        """Rotate the object based on mouse movement"""
        delta_x = -(event.x - self._last_mouse_x) * 0.005  # Adjust this scaling factor as needed
        delta_y = (event.y - self._last_mouse_y) * 0.005  # Adjust this scaling factor as needed
        delta_vector = np.array([delta_x, delta_y, 0])

        # Calculate the rotation angle and axis
        # np.linalg.norm computes the Euclidean norm sqrt(x^2+y^2+z^2)
        angle = np.linalg.norm(delta_vector)
        if angle != 0:
            # Axis: find axis of the rotation. (0, 0, -1) as fixed camera position.
            rot_axis = np.cross(np.array([0, 0, -1]), delta_vector)
            # normalize rot_axis
            rot_axis = rot_axis / np.linalg.norm(rot_axis) if np.linalg.norm(rot_axis) != 0 else rot_axis

            # Create the rotation quaternion
            rotation = axis_angle_to_quaternion(rot_axis, angle)

            # Apply the rotation to the previous orientation
            new_orientation = quaternion_multiply(rotation, self._start_orientation)
            self._geometry_handler.reset_rotation(new_orientation)

            # Convert the quaternion to Euler angles
            x_angle, y_angle, z_angle = quaternion_to_euler(new_orientation)

            # Update the sliders
            self._x_slider.set(x_angle)
            self._y_slider.set(y_angle)
            self._z_slider.set(z_angle)
            self.__changed()

    def __set_file(self, obj_file_path):
        self.__reset_rotation()
        with open(obj_file_path) as obj_file:
            self._geometry_handler.upload_object(*extract_data(obj_file))
