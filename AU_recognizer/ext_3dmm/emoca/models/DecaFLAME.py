import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as func

from ..utils.lbs import batch_rodrigues, vertices2landmarks, lbs


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful with extreme cases of euler angles like [0.0, pi, 0.0]
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the mesh and 2D/3D facial landmarks
    """

    # noinspection PyUnresolvedReferences
    def __init__(self, config):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(config.flame_model_path, 'rb') as f:
            # flame_model = Struct(**pickle.load(f, encoding='latin1'))
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.cfg = config
        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.n_shape], shapedirs[:, :, 300:300 + config.n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(config.flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.tensor(lmk_embeddings['static_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_embeddings['static_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx',
                             torch.tensor(lmk_embeddings['dynamic_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords',
                             torch.tensor(lmk_embeddings['dynamic_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.tensor(lmk_embeddings['full_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('full_lmk_bary_coords',
                             torch.tensor(lmk_embeddings['full_lmk_bary_coords'], dtype=self.dtype))

        neck_kin_chain = []
        neck_idx = 1
        curr_idx = torch.tensor(neck_idx, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    @staticmethod
    def _find_dynamic_lmk_idx_and_bcoords(pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    @staticmethod
    def _vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch. Tensor NxVx3, dtype = torch.float32
                    The Tensor of input vertices
                faces: torch.Tensor (N*F)x3, dtype = torch.Long
                    The faces of the mesh
                lmk_faces_idx: torch.Tensor N X L, dtype = torch.Long
                    The Tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.Tensor N X L X 3, dtype = torch.float32
                    The Tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.Tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(
            device=vertices.device) * num_verts

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def _vertices2landmarks2d(self, vertices, full_pose):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.Tensor NxVx3, dtype = torch.float32
                    The Tensor of input vertices
                full_pose: torch.Tensor N X 12, dtype = torch.float32
                    The Tensor with global pose, neck pose, jaw pose and eye pose (respectively) in axis angle format

            Returns:
                landmarks: torch.Tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size = vertices.shape[0]
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor,
                                         lmk_faces_idx,
                                         lmk_bary_coords)
        return landmarks2d

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                         self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if expression_params is None:
            expression_params = torch.zeros(batch_size, self.cfg.n_exp).to(shape_params.device)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype,
                          detach_pose_correctives=False)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor,
                                         lmk_faces_idx,
                                         lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(bz, 1),
                                         self.full_lmk_bary_coords.repeat(bz, 1, 1))

        return vertices, landmarks2d, landmarks3d


# noinspection PyPep8Naming
class FLAME_mediapipe(FLAME):

    def __init__(self, config):
        super().__init__(config)
        # static MEDIAPIPE landmark embeddings for FLAME
        lmk_embeddings_mediapipe = np.load(config.flame_mediapipe_lmk_embedding_path,
                                           allow_pickle=True, encoding='latin1')
        self.register_buffer('lmk_faces_idx_mediapipe',
                             torch.tensor(lmk_embeddings_mediapipe['lmk_face_idx'].astype(np.int64), dtype=torch.long))
        self.register_buffer('lmk_bary_coords_mediapipe',
                             torch.tensor(lmk_embeddings_mediapipe['lmk_b_coords'], dtype=self.dtype))

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        vertices, landmarks2d, landmarks3d = super().forward(shape_params, expression_params, pose_params,
                                                             eye_pose_params)
        batch_size = shape_params.shape[0]
        lmk_faces_idx_mediapipe = self.lmk_faces_idx_mediapipe.unsqueeze(dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords_mediapipe = self.lmk_bary_coords_mediapipe.unsqueeze(dim=0).expand(batch_size, -1,
                                                                                           -1).contiguous()
        landmarks2d_mediapipe = vertices2landmarks(vertices, self.faces_tensor,
                                                   lmk_faces_idx_mediapipe,
                                                   lmk_bary_coords_mediapipe)

        return vertices, landmarks2d, landmarks3d, landmarks2d_mediapipe  # , landmarks3d_mediapipe


class FLAMETex(nn.Module):
    """
    current FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/albedoModel2020_FLAME_albedoPart.npz'
    ## adapted from BFM
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/FLAME_albedo_from_BFM.npz'
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)

        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.

        else:
            print('texture type ', config.tex_type, 'not exist!')
            exit()

        n_tex = config.n_tex
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode):
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = func.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture
