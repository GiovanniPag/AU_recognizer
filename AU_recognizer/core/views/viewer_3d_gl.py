import platform as pf
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
from pyopengltk import OpenGLFrame

from AU_recognizer import VIEWER, FILL_COLOR, LINE_COLOR, CANVAS_COLOR, POINT_COLOR, POINT_SIZE, MOVING_STEP, GL_SOLID, \
    i18n, logger
from AU_recognizer.core.util import hex_to_float_rgba, nect_config, hex_to_float_rgb, \
    prepare_data_for_opengl
from AU_recognizer.core.util.geometry_3d import quaternion_to_matrix, axis_angle_to_quaternion, quaternion_multiply
from AU_recognizer.core.views import View, ComboLabel, CheckLabel, IconButton


class Viewer3DGl(View):
    def __init__(self, master, placeholder, obj_file_path=None):
        super().__init__(placeholder)
        self.master = master
        self.obj_file_path = obj_file_path
        self.viewer_frame = ttk.Frame(self)
        self.control_frame = ttk.Frame(self, style='Control.TFrame')
        self.canvas_3d = Frame3DGl(placeholder=self.viewer_frame, obj_file_path=obj_file_path)
        self.model_combobox = ComboLabel(master=self.control_frame, label_text="gl_combo",
                                         selected=i18n.gl_viewer["combo"][GL_SOLID],
                                         values=list(i18n.gl_viewer['combo'].values()), state="readonly")

        self.texture_checkbox = CheckLabel(master=self.control_frame, label_text="gl_texture", default=True,
                                           command=...)
        self.lighting_checkbox = CheckLabel(master=self.control_frame, label_text="gl_lightning", default=False,
                                            command=...)
        self.reset_button = IconButton(master=self.control_frame, asset_name="reset_view.png", tooltip="reset_view",
                                       command=...)
        self.settings_button = IconButton(master=self.control_frame, asset_name="settings.png", tooltip="open_set",
                                          command=...)

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
        self.model_combobox.create_view()
        self.model_combobox.grid(row=1, column=0, columnspan=3, sticky='e')
        # Checkbox for texture
        self.texture_checkbox.create_view()
        self.texture_checkbox.grid(row=2, column=0, columnspan=3, sticky='e')
        # Checkbox for lighting
        self.lighting_checkbox.create_view()
        self.lighting_checkbox.grid(row=3, column=0, columnspan=3, sticky='e')
        # Button to reset view
        self.reset_button.create_view()
        self.reset_button.grid(row=4, column=1, sticky='e')
        # Button to open settings
        self.settings_button.create_view()
        self.settings_button.grid(row=4, column=2, sticky='e')
        # FPS counter TODO: MOVE INSIDE VIEWER
        # self.fps_label = ttk.Label(self, text="FPS: 0")
        # self.fps_label.pack(side=tk.BOTTOM, anchor='se', padx=5, pady=5)

    def update_language(self):
        self.model_combobox.update_language(sel=i18n.gl_viewer["combo"][GL_SOLID],
                                            values=list(i18n.gl_viewer['combo'].values()))
        self.texture_checkbox.update_language()
        self.lighting_checkbox.update_language()
        self.reset_button.update_language()
        self.settings_button.update_language()

    def display(self, animate):
        self.canvas_3d.animate = animate


class Frame3DGl(OpenGLFrame):
    def __init__(self, placeholder, obj_file_path=None):
        super().__init__(placeholder)
        self._fill_color = hex_to_float_rgba(str(nect_config[VIEWER][FILL_COLOR]))
        self._line_color = hex_to_float_rgba(str(nect_config[VIEWER][LINE_COLOR]))
        self._canvas_color = hex_to_float_rgba(str(nect_config[VIEWER][CANVAS_COLOR]))
        self._moving_step = float(nect_config[VIEWER][MOVING_STEP])
        self._point_size = int(nect_config[VIEWER][POINT_SIZE])
        self._point_color = hex_to_float_rgb(str(nect_config[VIEWER][POINT_COLOR]))
        self._canvas_w = 800
        self._canvas_h = 400
        self._zoom = -5
        self._rotation = np.array([0, 0, 0, 1])  # Quaternion [x, y, z, w]
        self._position = [0, 0]  # x, y
        # mouse drag
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._is_left_mouse_button_down = False
        # mouse pan
        self._last_mouse_x_pan = 0
        self._last_mouse_y_pan = 0
        self._is_right_mouse_button_down = False
        # Placeholder for loaded model data
        self.model = None
        self.shader = None
        # default shader
        self.render_mode = 'solid'  # Default mode: 'solid', 'wireframe', or 'points'
        self.vertex_array_object = None
        self.proj = None
        self.textures = {}
        self._obj_file_path = Path(obj_file_path)
        # open obj file
        if obj_file_path:
            self.__set_file(self._obj_file_path)
        # bind events for mouse and keys
        self.bind_events()

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

        self.bind_all('<KeyPress-3>', lambda e: self.switch_render_mode('solid'))
        self.bind_all('<KeyPress-4>', lambda e: self.switch_render_mode('wireframe'))
        self.bind_all('<KeyPress-5>', lambda e: self.switch_render_mode('points'))

    def create_object(self, shader):
        vao_dict = {}
        for material, data in self.model['batched_data'].items():
            vertex_array_object = glGenVertexArrays(1)
            glBindVertexArray(vertex_array_object)
            # Generate vertex buffer
            vertex_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
            vertex_array = data['vertex_data']
            glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, GL_STATIC_DRAW)

            # Set shader uniforms based on mode
            shader.set_uniform("isWireframe", self.render_mode == 'wireframe')
            shader.set_uniform("isPoints", self.render_mode == 'points')

            # Handle textures, normal maps, or vertex color
            shader.set_uniform("useTexture", self.use_texture)
            shader.set_uniform("useNormalMap", self.use_normal_map)
            shader.set_uniform("useVertexColor", self.use_vertex_color)

            # Set colors from settings
            shader.set_uniform("solidColor", self.solid_color)
            shader.set_uniform("wireframeColor", self.wireframe_color)
            shader.set_uniform("pointsColor", self.points_color)

            # Set up position attribute
            position = glGetAttribLocation(shader, bytestr('position'))
            glEnableVertexAttribArray(position)
            glVertexAttribPointer(position, 3, GL_FLOAT, False, 5 * 4, ctypes.c_void_p(0))
            # Set up texture coordinate attribute
            texcoord = glGetAttribLocation(shader, bytestr('texcoord'))
            glEnableVertexAttribArray(texcoord)
            glVertexAttribPointer(texcoord, 2, GL_FLOAT, False, 5 * 4, ctypes.c_void_p(12))
            # Generate index buffer
            if data['index_data'].size > 0:
                element_buffer = glGenBuffers(1)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
                faces_array = data['index_data']
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces_array.nbytes, faces_array, GL_STATIC_DRAW)
            # Store the VAO per material
            vao_dict[material] = vertex_array_object
            # Unbind the VAO and buffers
            glBindVertexArray(0)
            glDisableVertexAttribArray(position)
            glDisableVertexAttribArray(texcoord)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        return vao_dict

    def initgl(self):
        self.shader = compileProgram(
            compileShader(vertex_shader_code, GL_VERTEX_SHADER),
            compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
        )
        self.vertex_array_object = self.create_object(self.shader)
        self.proj = glGetUniformLocation(self.shader, bytestr('proj'))
        # Set up OpenGL settings here
        glClearColor(*self._canvas_color)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CW)  # Assuming the faces are counter-clockwise in your model

        glEnable(GL_PROGRAM_POINT_SIZE)
        glColor3fv(self._point_color)
        glPointSize(self._point_size)

        # Load and bind textures
        self.load_textures()

    def switch_render_mode(self, mode):
        if mode in ['solid', 'wireframe', 'points']:
            self.render_mode = mode
        else:
            logger.error("Invalid render mode. Choose 'solid', 'wireframe', or 'points'.")

    def load_textures(self):
        texture_unit = GL_TEXTURE0
        glActiveTexture(texture_unit)

        for material_name, material in self.model['materials'].items():
            if 'texture' in material:
                texture_path = Path(material['texture'])
                texture_id = load_texture(texture_path)
                self.textures[material_name] = texture_id

    def redraw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        # Set rendering mode: solid, wireframe, or points
        if self.render_mode == 'solid':
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif self.render_mode == 'wireframe':
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        elif self.render_mode == 'points':
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)

        rotation_matrix = quaternion_to_matrix(self._rotation)
        model_matrix = np.identity(4)
        model_matrix[:3, :3] = rotation_matrix

        zoom_matrix = np.identity(4)
        zoom_matrix[3, 3] = 1 / (1 - self._zoom)

        translation_matrix = np.identity(4)
        translation_matrix[0, 3] = -self._position[0]
        translation_matrix[1, 3] = -self._position[1]

        projection_matrix = np.dot(zoom_matrix, np.dot(translation_matrix, model_matrix))

        glUniformMatrix4fv(self.proj, 1, GL_FALSE, projection_matrix.T)

        for material, vao in self.vertex_array_object.items():
            # Bind the VAO
            glBindVertexArray(vao)

            # Bind the appropriate texture for the material
            if material in self.textures and self.current_shader == "texture":
                glBindTexture(GL_TEXTURE_2D, self.textures[material])

            if 'index_data' in self.model['batched_data'][material]:
                glDrawElements(GL_TRIANGLES, len(self.model['batched_data'][material]['index_data']), GL_UNSIGNED_INT,
                               None)
            else:
                glDrawArrays(GL_POINTS, 0, len(self.model['batched_data'][material]['vertex_data']))
            glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)  # Unbind the VAO when done
        glUseProgram(0)

    def __set_file(self, obj_file_path):
        batched_data, materials = prepare_data_for_opengl(*extract_data(obj_file_path))
        self.model = {'batched_data': batched_data, 'materials': materials}

    def key_handler(self, event):
        if event.keysym == 'Up' or event.keysym == 'w':
            self._position[1] -= self._moving_step
        elif event.keysym == 'Down' or event.keysym == 's':
            self._position[1] += self._moving_step
        elif event.keysym == 'Left' or event.keysym == 'a':
            self._position[0] += self._moving_step
        elif event.keysym == 'Right' or event.keysym == 'd':
            self._position[0] -= self._moving_step

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
            self._zoom += delta * 0.1

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

        if self._is_right_mouse_button_down:
            # Panning logic, considering zoom for sensitivity
            dx = event.x - self._last_mouse_x_pan
            dy = event.y - self._last_mouse_y_pan

            # Zoom sensitivity adjustments (more sensitive when zoomed in, less when zoomed out)
            zoom_sensitivity = 0.005  # Base sensitivity factor
            zoom_factor = max(1, abs(self._zoom))  # Ensure zoom factor is never too small
            # Inverse logarithmic scaling to reverse the sensitivity relationship (more sensitive zoomed in)
            sensitivity = zoom_sensitivity / np.log(zoom_factor + 1)  # Inverse scaling (log(x+1))
            # Update position with adjusted sensitivity
            self._position[0] -= dx * sensitivity
            self._position[1] += dy * sensitivity
            # Update last mouse position
            self._last_mouse_x_pan = event.x
            self._last_mouse_y_pan = event.y

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
    img_data = np.array(image.convert("RGBA"), dtype=np.uint8)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

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
layout(location = 1) in vec2 inTexCoords;
layout(location = 2) in vec3 inVertexColor;
layout (location = 3) in vec3 aNormal;    // Geometric normal
layout(location = 4) in vec3 inTangent;  // Tangent vector
layout(location = 5) in vec3 inBitangent; // Bitangent vector

out vec2 fragTexCoords;
out vec3 fragColor;
out mat3 TBN; // Tangent-Bitangent-Normal matrix to convert normals to world space


uniform mat4 projectionMatrix; // Projection matrix (includes rotation, zoom, panning)


void main() {
    gl_Position = projectionMatrix * vec4(inPosition, 1.0);

    fragTexCoords = inTexCoords;
    fragColor = inVertexColor;
    fragTangent = inTangent;
    fragBitangent = inBitangent;
}

"""

# Fragment Shader for Solid Color / Per-Vertex Color
fragment_shader_code = """
#version 330 core

in vec2 fragTexCoords;
in vec3 fragColor;
in vec3 fragTangent;
in vec3 fragBitangent;

out vec4 FragColor;

uniform bool useTexture;        // Toggle texture usage
uniform bool useNormalMap;      // Toggle normal mapping
uniform sampler2D textureMap;   // Diffuse texture
uniform sampler2D normalMap;    // Normal map

uniform vec3 solidColor;        // Default solid color
uniform vec3 wireframeColor;    // Wireframe color
uniform vec3 pointsColor;       // Points color

uniform bool isWireframe;
uniform bool isPoints;

void main() {
    vec3 finalColor;

    if (isWireframe) {
        if (useVertexColor) {
            // Use vertex color
            finalColor = fragColor;
        } else {
            // Use wireframe color
            finalColor = wireframeColor;
        }
    } else if (isPoints) {
        if (useVertexColor) {
            // Use vertex color
            finalColor = fragColor;
        } else {
            // Use points color
            finalColor = pointsColor;
        }
    } else {
        if (useTexture) {
            // Apply texture mapping
            vec3 texColor = texture(textureMap, fragTexCoords).rgb;
            if (useNormalMap) {
                // Apply normal mapping if enabled
                vec3 normalMapNormal = texture(normalMap, fragTexCoords).rgb * 2.0 - 1.0;

                // Construct TBN matrix (Tangent, Bitangent, and default normal)
                vec3 N = vec3(0.0, 0.0, 1.0);  // Use default normal pointing up since there's no per-vertex normals
                mat3 TBN = mat3(normalize(fragTangent), normalize(fragBitangent), N);
            
                // Transform normal from tangent space to world space
                vec3 worldNormal = normalize(TBN * normalMapNormal);
                
                
                // Lighting and shading should go here
                finalColor = texColor;
            } else {
                finalColor = texColor;
            }
        } else {
            if (useVertexColor) {
                // Use vertex color
                finalColor = fragColor;
            } else {
                // Default to solid color
                finalColor = solidColor;
            }
        }
    }
    FragColor = vec4(finalColor, 1.0);
}
"""
