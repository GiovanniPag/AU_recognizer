import os

import cv2
import numpy as np
import torch
import torch.nn.functional as func


def upsample_mesh(vertices, normals, displacement_map, texture_map, dense_template):
    """ upsampling coarse mesh (with displacment map)
        vertices: vertices of coarse mesh, [nv, 3]
        normals: vertex normals, [nv, 3]
        faces: faces of coarse mesh, [nf, 3]
        texture_map: texture map, [256, 256, 3]
        displacement_map: displacment map, [256, 256]
        dense_template:
    Returns:
        dense_vertices: upsampled vertices with details, [number of dense vertices, 3]
        dense_colors: vertex color, [number of dense vertices, 3]
        dense_faces: [number of dense faces, 3]
    """
    dense_faces = dense_template['f']
    x_coords = dense_template['x_coords']
    y_coords = dense_template['y_coords']
    valid_pixel_ids = dense_template['valid_pixel_ids']
    valid_pixel_3d_faces = dense_template['valid_pixel_3d_faces']
    valid_pixel_b_coords = dense_template['valid_pixel_b_coords']
    pixel_3d_points = (vertices[valid_pixel_3d_faces[:, 0], :] *
                       valid_pixel_b_coords[:, 0][:, np.newaxis] +
                       vertices[valid_pixel_3d_faces[:, 1], :] *
                       valid_pixel_b_coords[:, 1][:, np.newaxis] +
                       vertices[valid_pixel_3d_faces[:, 2], :] *
                       valid_pixel_b_coords[:, 2][:, np.newaxis])
    vertex_normals_ = normals
    pixel_3d_normals = (vertex_normals_[valid_pixel_3d_faces[:, 0], :] *
                        valid_pixel_b_coords[:, 0][:, np.newaxis] +
                        vertex_normals_[valid_pixel_3d_faces[:, 1], :] *
                        valid_pixel_b_coords[:, 1][:, np.newaxis] +
                        vertex_normals_[valid_pixel_3d_faces[:, 2], :] *
                        valid_pixel_b_coords[:, 2][:, np.newaxis])
    pixel_3d_normals = pixel_3d_normals / np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
    displacements = displacement_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    dense_colors = texture_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    offsets = np.einsum('i,ij->ij', displacements, pixel_3d_normals)
    dense_vertices = pixel_3d_points + offsets
    return dense_vertices, dense_colors, dense_faces


# --------------------------------------- save obj
# copy from https://github.com/YadiraF/PRNet/blob/master/utils/write.py
def write_obj(obj_name,
              vertices,
              faces,
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=None,
              ):
    """ Save 3D face model with texture.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
        uvfaces
        inverse_face_order
        normal_map
    """
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.png')
    material_name = 'FaceTexture'
    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]
    # write obj
    with open(obj_name, 'w') as f:
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))
        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0],
                                                       colors[i, 1], colors[i, 2]))
        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i, 0], uvcoords[i, 1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    #  faces[i, 2], uvfaces[i, 2],
                    #  faces[i, 1], uvfaces[i, 1],
                    #  faces[i, 0], uvfaces[i, 0]
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            with open(mtl_name, 'w') as file:
                file.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name))  # map to image
                file.write(s)

                if normal_map is not None:
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    file.write(f'disp {normal_name}')
                    cv2.imwrite(
                        normal_name,
                        normal_map
                    )
            cv2.imwrite(texture_name, texture)


class C(object):
    pass


def dict2obj(d):
    if not isinstance(d, dict):
        return d

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def load_local_mask(image_size=256, mode='bbx'):
    regional_mask = None
    if mode == 'bbx':
        # UV space face attributes bbx in size 2048 (l r t b)
        face = np.array([400, 1648, 400, 1648])
        forehead = np.array([550, 1498, 430, 700 + 50])
        eye_nose = np.array([490, 1558, 700, 1050 + 50])
        mouth = np.array([574, 1474, 1050, 1550])
        ratio = image_size / 2048.
        face = (face * ratio).astype(np.int_)
        forehead = (forehead * ratio).astype(np.int_)
        eye_nose = (eye_nose * ratio).astype(np.int_)
        mouth = (mouth * ratio).astype(np.int_)
        regional_mask = np.array([face, forehead, eye_nose, mouth])
    return regional_mask


# ---------------------------- process/generate vertices, normals, faces
# Generates faces for a UV-mapped mesh. Each quadruple of neighboring pixels (2x2) is turned into two triangles
def generate_triangles(h, w):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    # .
    # w*h
    triangles = []
    margin = 0
    for x in range(margin, w - 1 - margin):
        for y in range(margin, h - 1 - margin):
            triangle0 = [y * w + x, y * w + x + 1, (y + 1) * w + x]
            triangle1 = [y * w + x + 1, (y + 1) * w + x + 1, (y + 1) * w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


# copy from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.linalg.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                          vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.linalg.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                          vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.linalg.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                          vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = func.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


# IO
def copy_state_dict(cur_state_dict, pre_state_dict, prefix='', load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                continue
            cur_state_dict[k].copy_(v)
        except Exception as err:
            print(err)
            continue


def batch_orth_proj(x, camera):
    """
        X is N x num_pquaternion_to_angle_axisoints x 3
    """
    camera = camera.clone().view(-1, 1, 3)
    x_trans = x[:, :, :2] + camera[:, :, 1:]
    x_trans = torch.cat([x_trans, x[:, :, 2:]], 2)
    # shape = x_trans.shape
    # xn = (camera[:, :, 0] * x_trans.view(shape[0], -1)).view(shape)
    xn = (camera[:, :, 0:1] * x_trans)
    return xn


# noinspection PyUnboundLocalVariable
def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, color='g', is_scale=True, rgb2bgr=True,
                         scale_colors=True):
    # visualize landmarks
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    if rgb2bgr:
        color_idx = [2, 1, 0]
    else:
        color_idx = [0, 1, 2]
    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, color_idx].copy()
        if scale_colors:
            image = image * 255
        if is_scale:
            predicted_landmark = predicted_landmarks[i] * image.shape[0] / 2 + image.shape[0] / 2
        else:
            predicted_landmark = predicted_landmarks[i]
        if predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks,
                                             gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2, 'r')
        else:
            image_landmarks = plot_verts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks,
                                             gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2, 'r')
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(
        vis_landmarks[:, :, :, color_idx].transpose(0, 3, 1, 2))
    if scale_colors:
        vis_landmarks /= 255.
    return vis_landmarks


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    return image.astype(np.uint8).copy()


# ---------------------------------- visualization
end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


# noinspection PyUnboundLocalVariable
def plot_kpts(image, kpts, color='r'):
    """ Draw 68 key points
    Args:
        image: the input image
        kpts: (68, 3).
        color
    """
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    for i in range(kpts.shape[0]):
        st = kpts[i, :2].astype(np.int32)
        if kpts.shape[1] == 4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        image = cv2.circle(image, (st[0], st[1]), 1, c, 2)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2].astype(np.int32)
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image


# noinspection PyUnboundLocalVariable
def plot_verts(image, kpts, color='r'):
    """ Draw 68 key points
    Args:
        image: the input image
        kpts: (68, 3).
        color
    """
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, c, 2)

    return image
