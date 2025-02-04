from pathlib import Path

import numpy as np

"""
@brief: Extracts vertices, faces, texture coordinates, and material information from an .obj file.

@param filepath: Path of a .obj file

@ret  : vertices (a numpy array of shape (n, 3) where n is the number of vertices)
@ret  : faces (a list of tuples, each containing face indices, texture coordinate indices, and material name)
@ret  : texcoords (a numpy array of texture coordinates)
@ret  : materials (a dictionary with material properties)
"""


class OBJ:
    def __init__(self, filepath=None, generate_normals=True):
        self.vertices = []  # 3D vertices (x, y, z)
        self.vertex_colors = []  # 3D vertices colors (r, g, b)
        self.texcoords = []  # 2D texture coordinates (u, v)
        self.normals = []  # 3D normals (nx, ny, nz)
        self.faces = []  # Faces will store vertex indices and texcoord indices and normal indices
        self.tangents = None
        self.materials = {}
        self.current_material = None
        self.file = None
        if filepath:
            self.load(filepath, generate_normals)

    def has_texture(self):
        return bool(self.texcoords)

    def set_vertex_colors(self, colors):
        """Assign colors to vertices."""
        if len(colors) != len(self.vertices):
            raise ValueError("Number of colors must match the number of vertices.")
        self.vertex_colors = colors

    def load(self, filepath, generate_normals=True):
        self.file = Path(filepath)
        has_normals = False
        self.current_material = None
        with self.file.open() as f:
            for line in f:
                if line.startswith('v '):  # Vertex position
                    # Split the line into parts
                    parts = list(map(float, line.split()[1:]))  # Convert to floats starting from index 1
                    # The first 3 components are vertex positions (x, y, z)
                    self.vertices.append(parts[:3])
                    # Check if RGB values exist (line has 6 values)
                    if len(parts) > 3:
                        # Next three values are RGB
                        self.vertex_colors.append(parts[3:])
                    else:
                        # If RGB is missing, assign a default color (e.g., white)
                        self.vertex_colors.append([1.0, 1.0, 1.0])
                elif line.startswith('vt '):  # Texture coordinate
                    self.texcoords.append(list(map(float, line[3:].split())))
                elif line.startswith('vn '):  # Vertex normal
                    self.normals.append(list(map(float, line[3:].split())))
                    has_normals = True
                elif line.startswith('mtllib '):  # Material library
                    mtl_files = line.split()[1:]
                    for mtl_file in mtl_files:
                        self.load_mtl(
                            Path(mtl_file) if Path(mtl_file).is_absolute() else (self.file.parent / mtl_file).resolve())
                elif line.startswith('usemtl '):  # Use material
                    self.current_material = line.split()[1]
                elif line.startswith('f '):  # Face
                    face_data = line[2:].split()
                    face_indices = []
                    texcoord_indices = []
                    normal_indices = []
                    for vertex in face_data:
                        parts = vertex.split('/')
                        face_indices.append(int(parts[0]) - 1)
                        texcoord_indices.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1)
                        normal_indices.append(int(parts[2]) - 1 if len(parts) > 2 and parts[2] else -1)
                    self.faces.append((face_indices, texcoord_indices, normal_indices, self.current_material))
        if generate_normals:
            if not has_normals:
                self.calculate_normals()
            self.calculate_tangents()

    def load_mtl(self, mtl_filepath):
        current_material = None
        with open(mtl_filepath) as mtl_file:
            for line in mtl_file:
                if line.startswith('newmtl '):  # New material
                    current_material = Material(line.split()[1])
                    self.materials[current_material.name] = current_material
                elif line.startswith('Ka ') and current_material:  # Ambient color
                    current_material.ambient = list(map(float, line.split()[1:4]))
                elif line.startswith('Kd ') and current_material:  # Diffuse color
                    current_material.diffuse = list(map(float, line.split()[1:4]))
                elif line.startswith('Ks ') and current_material:  # Specular color
                    current_material.specular = list(map(float, line.split()[1:4]))
                elif line.startswith('Ns ') and current_material:  # Shininess
                    current_material.shininess = float(line.split()[1])
                elif line.startswith('map_Kd ') and current_material:  # Diffuse texture map
                    f = line.split()[1]
                    current_material.texture_map = Path(f) if Path(f).is_absolute() else (
                            mtl_filepath.parent / f).resolve()
                elif line.startswith('disp ') and current_material:
                    f = line.split()[1]
                    current_material.normal_map = Path(f) if Path(f).is_absolute() else (
                            mtl_filepath.parent / f).resolve()

    def calculate_normals(self):
        # Initialize normals array
        normals_accum = np.zeros((len(self.vertices), 3), dtype=np.float32)
        self.normals = []  # Clear any existing normals

        for face_indices, t_i, n_i, mat in self.faces:
            v0, v1, v2 = [np.array(self.vertices[idx])[:3] for idx in face_indices]
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)  # Normalize the face normal
            for idx in face_indices:
                normals_accum[idx] += normal

        # Normalize the accumulated normals
        for i in range(len(normals_accum)):
            if np.linalg.norm(normals_accum[i]) > 0:
                normals_accum[i] /= np.linalg.norm(normals_accum[i])
            self.normals.append(normals_accum[i].tolist())
        # Update faces to include normal indices
        updated_faces = []
        for face_indices, t_i, n_i, mat in self.faces:
            normal_idx = [face_indices[i] for i in range(len(face_indices))]
            updated_faces.append((face_indices, t_i, normal_idx, mat))

        self.faces = updated_faces

    def calculate_tangents(self):
        self.tangents = np.zeros((len(self.vertices), 3), dtype=np.float32)

        for face_indices, texcoord_indices, n_idx, mat in self.faces:
            if len(texcoord_indices) < 3 or texcoord_indices[0] == -1:
                continue

            v0, v1, v2 = [np.array(self.vertices[idx]) for idx in face_indices]
            uv0, uv1, uv2 = [np.array(self.texcoords[idx]) for idx in texcoord_indices]

            delta_pos1 = v1 - v0
            delta_pos2 = v2 - v0
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0

            r = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0])
            tangent = (delta_pos1 * delta_uv2[1] - delta_pos2 * delta_uv1[1]) * r

            for idx in face_indices:
                self.tangents[idx] += tangent

        # Normalize tangents
        self.tangents = [t / np.linalg.norm(t) if np.linalg.norm(t) > 0 else t for t in self.tangents]

    def get_vertices(self):
        return np.array(self.vertices)

    def get_texcoords(self):
        return np.array(self.texcoords)

    def get_normals(self):
        return np.array(self.normals)

    def get_faces(self):
        return self.faces

    def get_faces_indices(self):
        return np.array([face[0] for face in self.faces], dtype=np.int32)

    def get_materials(self):
        return self.materials

    def get_material(self, material_name):
        return self.materials.get(material_name)

    def compute_mesh_size(self):
        # Assuming vertices is a numpy array of shape (n, 3)
        min_bound = np.min(self.vertices, axis=0)
        max_bound = np.max(self.vertices, axis=0)
        mesh_size = max_bound - min_bound
        return mesh_size

    def get_bounding_box_diagonal(self):
        mesh_size = self.compute_mesh_size()
        return np.linalg.norm(mesh_size)

    def prepare_opengl_data(self):
        """
        Prepare data to be sent to OpenGL for rendering.
        This function interleaves vertex attributes
        """
        interleaved_data = {}
        # Interleave vertices, colors, texcoords, and normals
        for face in self.faces:
            vertex_indices, texcoord_indices, n_idxs, material = face
            if material not in interleaved_data:
                interleaved_data[material] = {'vertex_data': [], 'index_data': []}

            vertex_data = interleaved_data[material]['vertex_data']
            index_data = interleaved_data[material]['index_data']

            for i, vertex_index in enumerate(vertex_indices):
                # Add vertex positions
                verts = []
                verts.extend(self.vertices[vertex_index])
                verts.extend(self.vertex_colors[vertex_index])
                # Add texture coordinates (if available)
                if texcoord_indices[i] != -1:
                    verts.extend(self.texcoords[texcoord_indices[i]])
                else:
                    verts.extend([-1, -1])  # Default to no texcoord
                verts.extend(self.normals[n_idxs[i]])
                verts.extend(self.tangents[vertex_index].tolist())
                # each verts array is [
                # position: x, y, z, color: r, g, b,
                # texture:  u, v,
                # normals: nx, ny, nz,
                # tangent: tx, ty, tz]
                vertex_data.append(verts)
                index_data.append(len(vertex_data) - 1)

        for material in interleaved_data:
            interleaved_data[material]['vertex_data'] = np.array(interleaved_data[material]['vertex_data'],
                                                                 dtype=np.float32)
            interleaved_data[material]['index_data'] = np.array(interleaved_data[material]['index_data'],
                                                                dtype=np.uint32)
        return interleaved_data, self.materials

    def save(self, output_path):
        """Save OBJ with vertex colors."""
        with open(output_path, 'w') as file:
            for i, vertex in enumerate(self.vertices):
                line = f"v {' '.join(map(str, vertex))}"
                line += f" {' '.join(map(str, self.vertex_colors[i]))}"
                file.write(line + '\n')

            for face in self.faces:
                face_indices, texcoord_indices, normal_indices, material = face
                face_str = ' '.join(
                    f"{vi + 1}"  # Add 1 to make it 1-indexed
                    for vi in face_indices  # Iterate over face_indices directly
                )
                file.write(f"f {face_str}\n")


class Material:
    def __init__(self, name):
        self.name = name
        self.ambient = [0.0, 0.0, 0.0]
        self.diffuse = [0.0, 0.0, 0.0]
        self.specular = [0.0, 0.0, 0.0]
        self.shininess = 0.0
        self.texture_map = None
        self.normal_map = None
