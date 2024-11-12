import platform as pf
import tkinter as tk
from pprint import pprint
from tkinter import ttk

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
from pyopengltk import OpenGLFrame

from AU_recognizer import VIEWER, FILL_COLOR, LINE_COLOR, CANVAS_COLOR, POINT_COLOR, POINT_SIZE, MOVING_STEP, GL_SOLID, \
    i18n, logger, GL_U_VALUE, GL_U_POINTER, SKY_COLOR, GROUND_COLOR, GL_U_TYPE, GL_DEFAULT, GL_NO, GL_WIREFRAME, \
    GL_V_COLOR, GL_NORMAL, GL_C_POINTS, GL_C_TEXTURE, GL_C_NORMAL_MAP
from AU_recognizer.core.util import nect_config, hex_to_float_rgb, hex_to_float_rgba
from AU_recognizer.core.util.OBJ import OBJ
from AU_recognizer.core.util.geometry_3d import axis_angle_to_quaternion, quaternion_multiply, look_at, perspective, \
    quaternion_to_matrix
from AU_recognizer.core.views import View, ComboLabel, CheckLabel, IconButton


class Viewer3DGl(View):
    def __init__(self, master, placeholder, obj_file_path=None):
        super().__init__(placeholder)
        self.master = master
        self.obj = OBJ(filepath=obj_file_path)
        self.viewer_frame = ttk.Frame(self)
        self.control_frame = ttk.Frame(self, style='Control.TFrame')
        self.canvas_3d = Frame3DGl(placeholder=self.viewer_frame, obj=self.obj)
        self.render_combobox = ComboLabel(master=self.control_frame, label_text="gl_combo",
                                          selected=i18n.gl_viewer["combo"][GL_SOLID],
                                          values=list(i18n.gl_viewer['combo'].values()), state="readonly")
        self.color_combobox = ComboLabel(master=self.control_frame, label_text="gl_color",
                                         selected=i18n.gl_viewer["color_combo"][GL_DEFAULT],
                                         values=list(i18n.gl_viewer['color_combo'].values()), state="readonly")
        self.normal_combobox = ComboLabel(master=self.control_frame, label_text="gl_normal",
                                          selected=i18n.gl_viewer["normal_combo"][GL_NO],
                                          values=list(i18n.gl_viewer['normal_combo'].values()), state="readonly")
        self.lighting_checkbox = CheckLabel(master=self.control_frame, label_text="gl_light", default=False,
                                            command=self._toggle_light)
        self.reset_button = IconButton(master=self.control_frame, asset_name="reset_view.png", tooltip="reset_view",
                                       command=self.canvas_3d.reset_view)
        self.settings_button = IconButton(master=self.control_frame, asset_name="settings.png", tooltip="open_set",
                                          command=...)

    def _toggle_light(self):
        self.canvas_3d.update_shader_uniforms([("useLight", self.lighting_checkbox.get_value())], start_shader=True)

    def _change_normal_mode(self, new_mode):
        if new_mode == i18n.gl_viewer["normal_combo"][GL_NO]:
            self.canvas_3d.update_shader_uniforms([("useNormalMap", False),
                                                   ("useNormal", False)], start_shader=True)
        if new_mode == i18n.gl_viewer["normal_combo"][GL_NORMAL]:
            self.canvas_3d.update_shader_uniforms([("useNormalMap", False),
                                                   ("useNormal", True)], start_shader=True)
        if new_mode == i18n.gl_viewer["normal_combo"][GL_C_NORMAL_MAP]:
            self.canvas_3d.update_shader_uniforms([("useNormalMap", True),
                                                   ("useNormal", False)], start_shader=True)

    def _change_color_mode(self, new_mode):
        if new_mode == i18n.gl_viewer["color_combo"][GL_DEFAULT]:
            self.canvas_3d.update_shader_uniforms([("useTexture", False),
                                                   ("useVertexColor", False)], start_shader=True)
        if new_mode == i18n.gl_viewer["color_combo"][GL_V_COLOR]:
            self.canvas_3d.update_shader_uniforms([("useTexture", False),
                                                   ("useVertexColor", True)], start_shader=True)
        if new_mode == i18n.gl_viewer["color_combo"][GL_C_TEXTURE]:
            self.canvas_3d.update_shader_uniforms([("useTexture", True),
                                                   ("useVertexColor", False)], start_shader=True)

    def _change_render_mode(self, new_mode):
        if new_mode == i18n.gl_viewer["combo"][GL_SOLID]:
            self.canvas_3d.update_shader_uniforms([("isPoints", False),
                                                   ("isWireframe", False), ], start_shader=True)
        if new_mode == i18n.gl_viewer["combo"][GL_WIREFRAME]:
            self.canvas_3d.update_shader_uniforms([("isPoints", False),
                                                   ("isWireframe", True), ], start_shader=True)
        if new_mode == i18n.gl_viewer["combo"][GL_C_POINTS]:
            self.canvas_3d.update_shader_uniforms([("isPoints", True),
                                                   ("isWireframe", False), ], start_shader=True)

    def create_view(self):
        self.rowconfigure(0, weight=7)  # make grid cell expandable
        self.rowconfigure(1, weight=1)  # make grid cell expandable
        self.columnconfigure(0, weight=1)
        self.viewer_frame.grid(row=0, column=0, sticky='nswe')
        self.control_frame.grid(row=1, column=0, sticky='nswe', padx=5, pady=5)
        self.control_frame.columnconfigure(0, weight=1)
        # grid canvas 3d
        self.canvas_3d.pack(fill=tk.BOTH, expand=True)
        # Modal box for view modes
        self.render_combobox.create_view()
        self.render_combobox.grid(row=1, column=0, columnspan=3, sticky='e')
        # Modal box for color
        self.color_combobox.create_view()
        self.color_combobox.grid(row=2, column=0, columnspan=3, sticky='e')
        # Modal box for normals
        self.normal_combobox.create_view()
        self.normal_combobox.grid(row=3, column=0, columnspan=3, sticky='e')
        # Checkbox for lighting
        self.lighting_checkbox.create_view()
        self.lighting_checkbox.grid(row=4, column=0, columnspan=3, sticky='e')
        # Button to reset view
        self.reset_button.create_view()
        self.reset_button.grid(row=5, column=1, sticky='e')
        # Button to open settings
        self.settings_button.create_view()
        self.settings_button.grid(row=5, column=2, sticky='e')

        self.render_combobox.bind_combobox_event(self._change_render_mode)
        self.color_combobox.bind_combobox_event(self._change_color_mode)
        self.normal_combobox.bind_combobox_event(self._change_normal_mode)

        # FPS counter TODO: MOVE INSIDE VIEWER
        # self.fps_label = ttk.Label(self, text="FPS: 0")
        # self.fps_label.pack(side=tk.BOTTOM, anchor='se', padx=5, pady=5)

    def update_language(self):
        self.render_combobox.update_language(sel=i18n.gl_viewer["combo"][GL_SOLID],
                                             values=list(i18n.gl_viewer['combo'].values()))
        self.color_combobox.update_language(sel=i18n.gl_viewer["color_combo"][GL_DEFAULT],
                                            values=list(i18n.gl_viewer['color_combo'].values()))
        self.normal_combobox.update_language(sel=i18n.gl_viewer["normal_combo"][GL_NO],
                                             values=list(i18n.gl_viewer['normal_combo'].values()))
        self.lighting_checkbox.update_language()
        self.reset_button.update_language()
        self.settings_button.update_language()

    def display(self, animate):
        self.canvas_3d.animate = animate


class Frame3DGl(OpenGLFrame):
    def __init__(self, placeholder, obj: OBJ = None):
        super().__init__(placeholder)
        self._canvas_color = hex_to_float_rgba(str(nect_config[VIEWER][CANVAS_COLOR]))
        self._moving_step = float(nect_config[VIEWER][MOVING_STEP])
        self._point_size = int(nect_config[VIEWER][POINT_SIZE])
        self.shader_uniform = {
            "model": {
                GL_U_TYPE: "mat4",
                GL_U_POINTER: None,
                GL_U_VALUE: None,
            },
            "view": {
                GL_U_TYPE: "mat4",
                GL_U_POINTER: None,
                GL_U_VALUE: None,
            },
            "proj": {
                GL_U_TYPE: "mat4",
                GL_U_POINTER: None,
                GL_U_VALUE: None,
            },
            "solidColor": {
                GL_U_TYPE: "vec3",
                GL_U_POINTER: None,
                GL_U_VALUE: hex_to_float_rgb(str(nect_config[VIEWER][FILL_COLOR])),
            },
            "wireframeColor": {
                GL_U_TYPE: "vec3",
                GL_U_POINTER: None,
                GL_U_VALUE: hex_to_float_rgb(str(nect_config[VIEWER][LINE_COLOR])),
            },
            "pointsColor": {
                GL_U_TYPE: "vec3",
                GL_U_POINTER: None,
                GL_U_VALUE: hex_to_float_rgb(str(nect_config[VIEWER][POINT_COLOR])),
            },
            "useTexture": {
                GL_U_TYPE: "bool",
                GL_U_POINTER: None,
                GL_U_VALUE: False,
            },
            "useNormalMap": {
                GL_U_TYPE: "bool",
                GL_U_POINTER: None,
                GL_U_VALUE: False,
            },
            "useNormal": {
                GL_U_TYPE: "bool",
                GL_U_POINTER: None,
                GL_U_VALUE: False,
            },
            "useLight": {
                GL_U_TYPE: "bool",
                GL_U_POINTER: None,
                GL_U_VALUE: False,
            },
            "isPoints": {
                GL_U_TYPE: "bool",
                GL_U_POINTER: None,
                GL_U_VALUE: False,
            },
            "isWireframe": {
                GL_U_TYPE: "bool",
                GL_U_POINTER: None,
                GL_U_VALUE: False,
            },
            "useVertexColor": {
                GL_U_TYPE: "bool",
                GL_U_POINTER: None,
                GL_U_VALUE: False,
            },
            "skyColor": {
                GL_U_TYPE: "vec3",
                GL_U_POINTER: None,
                GL_U_VALUE: hex_to_float_rgb(str(nect_config[VIEWER][SKY_COLOR])),
            },
            "groundColor": {
                GL_U_TYPE: "vec3",
                GL_U_POINTER: None,
                GL_U_VALUE: hex_to_float_rgb(str(nect_config[VIEWER][GROUND_COLOR])),
            },
        }
        self._canvas_w = 800
        self._canvas_h = 400
        self._rotation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion [x, y, z, w]
        self.camera_position = np.array([0.0, 0.0, 10.0])
        self.panning = np.array([0.0, 0.0])
        self.scale = 1.0
        # mouse drag
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._is_left_mouse_button_down = False
        # mouse pan
        self._last_mouse_x_pan = 0
        self._last_mouse_y_pan = 0
        self._is_right_mouse_button_down = False
        # Placeholder for loaded model data
        self.shader = None
        self.vertex_array_object = None
        self.textures = {}
        self.normal_maps = {}
        # open obj file
        self.obj = obj
        batched_data, materials = self.obj.prepare_opengl_data()
        self.model = {'batched_data': batched_data, 'materials': materials}
        # bind events for mouse and keys
        self.bind_events()

    def reset_view(self):
        self._rotation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion [x, y, z, w]
        self.camera_position = np.array([0.0, 0.0, 10.0])
        self.panning = np.array([0.0, 0.0])
        self.scale = 1.0

    def update_shader_uniforms(self, uniform_list, start_shader=False):
        if self.shader is not None:
            if start_shader:
                glUseProgram(self.shader)  # Activate the shader program
            for uniform in uniform_list:
                if isinstance(uniform, tuple) and len(uniform) == 2:
                    uniform_name, uniform_value = uniform
                    self.set_uniform_value(uniform_name, uniform_value)
                    self.update_shader_uniform(uniform_name)
                elif isinstance(uniform, str):
                    uniform_name = uniform
                    self.update_shader_uniform(uniform_name)
            if start_shader:
                glUseProgram(0)  # Deactivate the shader program
        else:
            logger.error("No shader program available")

    def update_shader_uniform(self, uniform_name):
        if uniform_name not in self.shader_uniform:
            logger.error(f"No uniform with name '{uniform_name}'")
            return

        uniform = self.shader_uniform[uniform_name]
        if uniform[GL_U_POINTER] is None:
            uniform[GL_U_POINTER] = glGetUniformLocation(self.shader, bytestr(uniform_name))
            if uniform[GL_U_POINTER] == -1:
                logger.error(f"Uniform '{uniform_name}' not found in shader.")
                return
        # Get the value and type
        value = uniform[GL_U_VALUE]
        uniform_type = uniform[GL_U_TYPE]

        # Call the appropriate OpenGL function based on the type
        if uniform_type == "mat4":
            glUniformMatrix4fv(uniform[GL_U_POINTER], 1, GL_FALSE, value)
        elif uniform_type == "vec3":
            glUniform3fv(uniform[GL_U_POINTER], 1, value)
        elif uniform_type == "bool":
            glUniform1i(uniform[GL_U_POINTER], int(value))
        else:
            logger.error(f"Uniform type '{uniform_type}' for '{uniform_name}' is not supported.")

    def set_uniform_value(self, uniform_name, new_value):
        if uniform_name not in self.shader_uniform:
            logger.error(f"No uniform with name '{uniform_name}'")
            return

        # Update the uniform value
        self.shader_uniform[uniform_name][GL_U_VALUE] = new_value

    def bind_events(self):
        # Catch the canvas resize event
        self.bind("<Configure>", self.__resized)
        # Bind the scrolling on canvas event to zoom slider (Button-4/5 on linux)
        self.bind('<Button-4>', self.mouse_wheel_handler)
        self.bind('<Button-5>', self.mouse_wheel_handler)
        # zoom for Windows and MacOS, but not Linux
        self.bind('<MouseWheel>', self.mouse_wheel_handler)
        # Bind keys for movement
        self.bind_all('<KeyPress-w>', self.key_handler)
        self.bind_all('<KeyPress-s>', self.key_handler)
        self.bind_all('<KeyPress-a>', self.key_handler)
        self.bind_all('<KeyPress-d>', self.key_handler)
        self.bind_all('<KeyPress-Up>', self.key_handler)
        self.bind_all('<KeyPress-Down>', self.key_handler)
        self.bind_all('<KeyPress-Left>', self.key_handler)
        self.bind_all('<KeyPress-Right>', self.key_handler)

        self.bind("<Button-1>", self.__start_rotate)  # Left click for rotation
        self.bind("<Button-3>", self.__start_pan)  # Right click for panning
        self.bind("<B1-Motion>", self.__on_mouse_move)  # Left drag for rotation
        self.bind("<B3-Motion>", self.__on_mouse_move)  # Right drag for panning
        self.bind("<ButtonRelease-1>", self.__end_rotate)
        self.bind("<ButtonRelease-3>", self.__end_pan)

    def create_object(self):
        vao_dict = {}
        for material_name, data in self.model['batched_data'].items():
            vertex_array_object = glGenVertexArrays(1)
            glBindVertexArray(vertex_array_object)
            # Generate vertex buffer
            vertex_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
            vertex_array = data['vertex_data']
            glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, GL_STATIC_DRAW)

            stride = (3 + 3 + 2 + 3 + 3) * ctypes.sizeof(
                ctypes.c_float)  # Position + Color + TexCoord + Normal + tangent +
            # Enable and define the vertex position attribute (location = 0)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))

            # Enable and define the vertex color attribute (location = 1)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))

            # Enable and define the texture coordinate attribute (location = 2)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, False, stride, ctypes.c_void_p(6 * ctypes.sizeof(ctypes.c_float)))

            # Enable and define the normal attribute (location = 3)
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 3, GL_FLOAT, False, stride, ctypes.c_void_p(8 * ctypes.sizeof(ctypes.c_float)))

            # Enable and define the tangent attribute (location = 4)
            glEnableVertexAttribArray(4)
            glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(11 * ctypes.sizeof(ctypes.c_float)))

            # Generate index buffer
            if data['index_data'] is not None and data['index_data'].size > 0:
                element_buffer = glGenBuffers(1)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
                faces_array = data['index_data']
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces_array.nbytes, faces_array, GL_STATIC_DRAW)
            # Store the VAO per material
            vao_dict[material_name] = vertex_array_object
            # Unbind the VAO and buffers
            glBindVertexArray(0)
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)
            glDisableVertexAttribArray(2)
            glDisableVertexAttribArray(3)
            glDisableVertexAttribArray(4)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        return vao_dict

    def initgl(self):
        self.shader = compileProgram(
            compileShader(vertex_shader_code, GL_VERTEX_SHADER),
            compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
        )

        glUseProgram(self.shader)  # Activate the shader program
        glUniform1i(glGetUniformLocation(self.shader, bytestr("textureMap")), 0)  # GL_TEXTURE0
        glUniform1i(glGetUniformLocation(self.shader, bytestr("normalMap")), 1)  # GL_TEXTURE1
        self.update_shader_uniforms(["solidColor",
                                     "wireframeColor",
                                     "pointsColor",
                                     "useTexture",
                                     "useNormalMap",
                                     "useNormal",
                                     "useLight",
                                     "isPoints",
                                     "isWireframe",
                                     "useVertexColor",
                                     "skyColor",
                                     "groundColor"
                                     ])
        glUseProgram(0)  # Deactivate the shader program
        self.vertex_array_object = self.create_object()
        # Set up OpenGL settings here
        glClearColor(*self._canvas_color)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CW)  # Assuming the faces are counter-clockwise
        glEnable(GL_PROGRAM_POINT_SIZE)
        glPointSize(self._point_size)
        # Load and bind textures
        self.load_asset()

    def load_asset(self):
        for material_name, material in self.model['materials'].items():
            if material.texture_map is not None:
                glActiveTexture(GL_TEXTURE0)
                texture_id = load_texture(material.texture_map)
                self.textures[material_name] = texture_id
            if material.normal_map is not None:
                glActiveTexture(GL_TEXTURE1)
                texture_id = load_texture(material.normal_map)
                self.normal_maps[material_name] = texture_id

    def redraw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        # Set rendering mode: solid, wireframe, or points
        if self.shader_uniform['isWireframe'][GL_U_VALUE]:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        elif self.shader_uniform['isPoints'][GL_U_VALUE]:
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Set up the projection matrix (perspective projection)
        aspect_ratio = self._canvas_w / self._canvas_h
        proj = perspective(45.0, aspect_ratio, 0.1, 100.0)
        # View matrix (camera)
        view = look_at(eye=self.camera_position, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))

        # Model matrix (rotation)
        model = np.eye(4)
        rot_matrix = quaternion_to_matrix(self._rotation)
        model[:3, :3] = rot_matrix[:3, :3]
        model = np.dot(np.array([
            [1, 0, 0, self.panning[0]],
            [0, 1, 0, self.panning[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]), model)
        # Apply scaling last
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = self.scale  # Uniform scale factor for x-axis
        scale_matrix[1, 1] = self.scale  # Uniform scale factor for y-axis
        scale_matrix[2, 2] = self.scale  # Uniform scale factor for z-axis
        # Apply scaling to the model
        model = np.dot(scale_matrix, model)

        self.update_shader_uniforms([("view", view.T),
                                     ("proj", proj.T),
                                     ("model", model.T)])

        for material_name, vao in self.vertex_array_object.items():
            # Bind the VAO
            glBindVertexArray(vao)

            # Bind the appropriate texture for the material
            if material_name in self.normal_maps and self.shader_uniform["useNormalMap"][GL_U_VALUE] is True:
                glActiveTexture(GL_TEXTURE1)  # Set to use texture unit 1 for the normal map
                glBindTexture(GL_TEXTURE_2D, self.normal_maps[material_name])

            # Bind the appropriate texture for the material
            if material_name in self.textures and self.shader_uniform["useTexture"][GL_U_VALUE] is True:
                glActiveTexture(GL_TEXTURE0)  # Set to use texture unit 0 for the diffuse texture
                glBindTexture(GL_TEXTURE_2D, self.textures[material_name])

            if 'index_data' in self.model['batched_data'][material_name] and self.model['batched_data'][material_name][
                'index_data'] is not None:
                glDrawElements(GL_TRIANGLES, len(self.model['batched_data'][material_name]['index_data']),
                               GL_UNSIGNED_INT,
                               None)
            else:
                glDrawArrays(GL_POINTS, 0, len(self.model['batched_data'][material_name]['vertex_data']))
            glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)  # Unbind the VAO when done
        glUseProgram(0)

    def key_handler(self, event):
        if event.keysym == 'Up' or event.keysym == 'w':
            self.panning[1] += self._moving_step
        elif event.keysym == 'Down' or event.keysym == 's':
            self.panning[1] -= self._moving_step
        elif event.keysym == 'Left' or event.keysym == 'a':
            self.panning[0] += self._moving_step
        elif event.keysym == 'Right' or event.keysym == 'd':
            self.panning[0] -= self._moving_step

    def mouse_wheel_handler(self, event):
        """Cross-platform mouse wheel scroll event for zooming"""
        if pf.system() == 'Windows':
            delta = event.delta / 120
        elif pf.system() == 'Darwin':
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
            self.camera_position[2] += delta * 0.1

    def __resized(self, *args):
        """Callback to the window resize events"""
        w, h = self.winfo_width(), self.winfo_height()
        if self._canvas_w != w or self._canvas_h != h:
            # Keep the object in the middle of the canvas
            self._canvas_w = w
            self._canvas_h = h
            glViewport(0, 0, w, h)

    def __start_rotate(self, event):
        """Start capturing the initial mouse click position for rotation"""
        self._is_left_mouse_button_down = True
        self._start_orientation = self._rotation
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

    def __start_pan(self, event):
        """Start panning on right-click"""
        self._is_right_mouse_button_down = True
        self._last_mouse_x_pan = event.x
        self._last_mouse_y_pan = event.y

    def __on_mouse_move(self, event):
        """Handle both rotation and panning when mouse moves"""
        if self._is_left_mouse_button_down:
            self._handle_rotation(event)
        if self._is_right_mouse_button_down:
            self._handle_panning(event)

    def _handle_panning(self, event):
        # Panning logic
        dx = event.x - self._last_mouse_x_pan
        dy = event.y - self._last_mouse_y_pan
        sensitivity = 0.01
        # Update position with adjusted sensitivity
        self.panning[0] -= dx * sensitivity
        self.panning[1] -= dy * sensitivity
        # Update last mouse position
        self._last_mouse_x_pan = event.x
        self._last_mouse_y_pan = event.y

    def _handle_rotation(self, event):
        # Rotation logic (Y-axis and X-axis rotation)
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
            self._rotation = new_orientation

    def __end_rotate(self, event):
        """End rotation on left-click release"""
        self._is_left_mouse_button_down = False

    def __end_pan(self, event):
        """End panning on right-click release"""
        self._is_right_mouse_button_down = False


# Avoiding glitches in pyopengl-3.0.x and python3.4
def bytestr(s):
    return s.encode("utf-8") + b"\000"


# Avoiding glitches in pyopengl-3.0.x and python3.4
def compileShader(source, shaderType):
    """
    Compile shader source of given type
        source -- GLSL source-code for the shader
    shaderType -- GLenum GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc,
        returns GLuint compiled shader reference
    raises RuntimeError when a compilation failure occurs
    """
    if isinstance(source, str):
        source = [source]
    elif isinstance(source, bytes):
        source = [source.decode('utf-8')]

    shader = glCreateShader(shaderType)
    glShaderSource(shader, source)
    glCompileShader(shader)
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        # this will be wrong if the user has
        # disabled traditional unpacking array support.
        raise RuntimeError(
            """Shader compile failure (%s): %s""" % (
                result,
                glGetShaderInfoLog(shader),
            ),
            source,
            shaderType,
        )
    return shader


def load_texture(texture_file):
    image = Image.open(texture_file)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip image vertically for OpenGL
    img_data = np.array(image.convert("RGB"), dtype=np.uint8)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


# The vertex shader code
vertex_shader_code = """
#version 330 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inVertexColor;
layout(location = 2) in vec2 inTexCoords;
layout(location = 3) in vec3 aNormal;    // Geometric normal
layout(location = 4) in vec3 aTangent;  // Tangent vector

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec2 fragTexCoords;
out vec3 fragColor;
out mat3 TBN;
out vec3 fragNormal;

void main() {
    gl_Position = proj * view * model * vec4(inPosition, 1.0);

    fragTexCoords = inTexCoords;
    fragColor = inVertexColor;
    fragNormal = normalize(mat3(transpose(inverse(model))) * aNormal);
    
    vec3 T = normalize(vec3(model * vec4(aTangent, 0.0)));
    vec3 N = normalize(vec3(model * vec4(aNormal, 0.0)));
    // re-orthogonalize T with respect to N
    T = normalize(T - dot(T, N) * N);
    // then retrieve perpendicular vector B with the cross product of T and N
    vec3 B = cross(N, T);
    TBN = mat3(T, B, N);
}

"""

# Fragment Shader for Solid Color / Per-Vertex Color
fragment_shader_code = """
#version 330 core

in vec2 fragTexCoords;
in vec3 fragColor;
in mat3 TBN;
in vec3 fragNormal;

out vec4 FragColor;

uniform bool useTexture;        // Toggle texture usage
uniform bool useNormalMap;      // Toggle normal mapping
uniform sampler2D textureMap;   // Diffuse texture
uniform sampler2D normalMap;    // Normal map

uniform vec3 solidColor;        // Default solid color
uniform vec3 wireframeColor;    // Wireframe color
uniform vec3 pointsColor;       // Points color

uniform bool useLight;          // Toggle normal mapping
uniform vec3 skyColor;    // Light color from above 
uniform vec3 groundColor; // Light color from below

uniform bool isWireframe;
uniform bool isPoints;
uniform bool useVertexColor;
uniform bool useNormal;      // Toggle normal mapping

vec3 applyLighting(vec3 color, vec3 normal) {
    float blendFactor = normal.y * 0.5 + 0.5;
    // Mix between sky color and ground color
    vec3 hemisphericLight = mix(groundColor, skyColor, blendFactor);
    return color * hemisphericLight;
}

vec3 calculateColor() {
    if (isWireframe) {
        return useVertexColor ? fragColor : wireframeColor;
    } else if (isPoints) {
        return useVertexColor ? fragColor : pointsColor;
    } else if (useTexture) {
        return texture(textureMap, fragTexCoords).rgb;
    } else {
        return useVertexColor ? fragColor : solidColor;
    }
}

void main() {
    vec3 finalColor = calculateColor();
    vec3 normal = fragNormal;
    if (useNormalMap) {
        // Apply normal mapping if enabled
        normal = texture(normalMap, fragTexCoords).rgb * 2.0 - 1.0;
        // Tangent space transformation
        normal = normalize(TBN * normal);  // Transform to world space
    }
    if(useLight){
        finalColor = applyLighting(finalColor, normal);
    }else if(useNormal){
        vec3 normalEffect = 0.5 + 0.5 * normal;  // Map normal from [-1,1] to [0,1]
        finalColor *= normalEffect;               // Blend with base color
    }
    FragColor = vec4(finalColor, 1.0);
}
"""
