import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
# noinspection PyPep8Naming
from ..utils import DecaUtils as util


class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        # pix_to_face(n,h,w,k), bary_coords(n,h,w,k,3),attribute: (n, nf, 3, d)
        vismask = (pix_to_face > -1).float()
        d = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        n, h, w, k, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(n * h * w * k, 1, 1).expand(n * h * w * k, 3, d)
        pixel_face_vals = attributes.gather(0, idx).view(n, h, w, k, 3, d)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class SRenderY(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
        # faces
        dense_triangles = util.generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None, :, :])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)
        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)
        # SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * 3 * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * 3 * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * 3 * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point'):
        """
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], rnage:[-1,1], projected vertices, in image space, for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        """
        batch_size = vertices.shape[0]
        # rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10
        # attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1),
                                transformed_face_normals.detach(),
                                face_vertices.detach(),
                                face_normals],
                               -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = func.grid_sample(albedos, grid, align_corners=False)
        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()
        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type == 'point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                      lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
            images = albedo_images * shading_images
        else:
            images = albedo_images
            shading_images = images.detach() * 0.
        outputs = {
            'images': images * alpha_images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images,
            'transformed_normals': transformed_normals,
        }
        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        """
            sh_coeff: [bz, 9, 3]
        """
        n = normal_images
        sh = torch.stack([
            n[:, 0] * 0. + 1., n[:, 0], n[:, 1],
            n[:, 2], n[:, 0] * n[:, 1], n[:, 0] * n[:, 2],
            n[:, 1] * n[:, 2], n[:, 0] ** 2 - n[:, 1] ** 2, 3 * (n[:, 2] ** 2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    @staticmethod
    def add_pointlight(vertices, normals, lights):
        """
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_positions = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = func.normalize(light_positions[:, :, None, :] - vertices[:, None, :, :], dim=3)
        normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    @staticmethod
    def add_directionlight(normals, lights):
        """
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_direction = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = func.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1),
                                              dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def render_shape(self, vertices, transformed_vertices, images=None, detail_normal_images=None, lights=None):
        """
        -- rendering shape with detail normal map
        """
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor(
                [
                    [-1, 1, 1],
                    [1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [0, 0, 1]
                ]
            )[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float() * 1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_colors.expand(batch_size, -1, -1, -1),
                                transformed_face_normals.detach(),
                                face_vertices.detach(),
                                face_normals],
                               -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2],
                                          albedo_images.shape[3], 3]).permute(0, 3, 1, 2).contiguous()
        shaded_images = albedo_images * shading_images

        if images is None:
            shape_images = shaded_images * alpha_images + torch.zeros_like(shaded_images).to(vertices.device) * (
                    1 - alpha_images)
        else:
            shape_images = shaded_images * alpha_images + images * (1 - alpha_images)
        return shape_images

    def render_depth(self, transformed_vertices):
        """
        -- rendering depth
        """
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3)
        z = z - z.min()
        z = z / z.max()
        # Attributes
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        depth_images = rendering[:, :1, :, :]
        return depth_images

    def render_normal(self, transformed_vertices, normals):
        """
        -- rendering normal
        """
        batch_size = normals.shape[0]

        # Attributes
        attributes = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        normal_images = rendering[:, :3, :, :]
        return normal_images

    def world2uv(self, vertices):
        """
        project vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        """
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices
