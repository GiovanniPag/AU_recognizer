import platform as pf
import time
from pathlib import Path

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
from pyopengltk import OpenGLFrame

from AU_recognizer import VIEWER, FILL_COLOR, LINE_COLOR, CANVAS_COLOR, POINT_COLOR, POINT_SIZE, MOVING_STEP
from AU_recognizer.core.util import hex_to_float_rgba, nect_config, extract_data, hex_to_float_rgb, load_materials, \
    prepare_data_for_opengl
from AU_recognizer.core.util.geometry_3d import quaternion_to_matrix, axis_angle_to_quaternion, quaternion_multiply


class Viewer3DGl(OpenGLFrame):
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
        # Placeholder for loaded model data
        self.model = None
        self.shader = None
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

        # Bind mouse events for rotation
        self.bind("<Button-1>", self.__start_rotate)
        self.bind("<B1-Motion>", self.__rotate)

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
            compileShader(vertex_texture_shader, GL_VERTEX_SHADER),
            compileShader(fragment_texture_shader, GL_FRAGMENT_SHADER)
        )
        self.vertex_array_object = self.create_object(self.shader)
        self.proj = glGetUniformLocation(self.shader, bytestr('proj'))

        # Set up OpenGL settings here
        glClearColor(*self._canvas_color)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glColor3fv(self._point_color)
        glPointSize(self._point_size)

        # Load and bind textures
        self.load_textures()

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

        rotation_matrix = quaternion_to_matrix(self._rotation)
        zoom_matrix = np.identity(4)
        zoom_matrix[3, 3] = 1 / (1 - self._zoom)

        model_matrix = np.identity(4)
        model_matrix[:3, :3] = rotation_matrix

        translation_matrix = np.identity(4)
        translation_matrix[0, 3] = -self._position[0]
        translation_matrix[1, 3] = -self._position[1]

        projection_matrix = np.dot(zoom_matrix, np.dot(translation_matrix, model_matrix))

        glUniformMatrix4fv(self.proj, 1, GL_FALSE, projection_matrix.T)

        for material, vao in self.vertex_array_object.items():
            # Bind the VAO
            glBindVertexArray(vao)

            # Bind the appropriate texture for the material
            if material in self.textures:
                glBindTexture(GL_TEXTURE_2D, self.textures[material])

            if 'index_data' in self.model['batched_data'][material]:
                glDrawElements(GL_TRIANGLES, len(self.model['batched_data'][material]['index_data']), GL_UNSIGNED_INT,
                               None)
            else:
                glDrawArrays(GL_POINTS, 0, len(self.model['batched_data'][material]['vertex_data']))
            glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)  # Unbind the VAO when done
        glUseProgram(0)
        # Set polygon mode to GL_LINE to draw only the edges of the polygons
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        # if 'faces' in self.model:
        #     glDrawElements(GL_TRIANGLES, len(self.model['faces']), GL_UNSIGNED_INT, None)
        # else:
        #     # Fallback to draw using vertices
        #     glDrawArrays(GL_POINTS, 0, len(self.model['vertices']))

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
        self._start_orientation = self._rotation
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
            self._rotation = new_orientation


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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


vertex_texture_shader = """#version 130 
in vec3 position;
in vec2 texcoord;  // Add texture coordinate input
out vec2 TexCoord; // Pass texture coordinate to fragment shader
uniform mat4 proj;
void main()
{
   gl_Position = proj * vec4(position, 1.0);
   TexCoord = texcoord; // Assign the input to the output
}
"""

vertex_color_shader = """#version 130 
in vec3 position;
varying vec3 vertex_color;
uniform mat4 proj;
void main()
{
   gl_Position = proj * vec4(position, 1.0);
   gl_PointSize = 4.0 / (0.5 + length(position));
   vertex_color = vec3( position.x / 2 + 0.5, position.y / 2 + 0.5, position.z / 2 + 0.5);
}
"""

fragment_texture_shader = """#version 130
in vec2 TexCoord;
uniform sampler2D ourTexture; // Texture sampler
void main()
{
   gl_FragColor = texture(ourTexture, TexCoord);
}
"""

fragment_color_shader = """#version 130
varying vec3 vertex_color;
void main()
{
   gl_FragColor = vec4(vertex_color,0.25f);
}
"""
