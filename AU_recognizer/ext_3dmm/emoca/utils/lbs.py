import torch
import torch.nn.functional as func


def blend_shapes(betas, shape_disps):
    """ Calculates the per vertex displacement due to the blend shapes
    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def vertices2joints(j_regressor, vertices):
    """ Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    j_regressor : torch.Tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices.
    vertices : torch.Tensor BxVx3
        The Tensor of mesh vertices

    Returns
    -------
    torch.Tensor BxJx3
        The location of the joints
    """

    return torch.einsum('bik,ji->bjk', [vertices, j_regressor])


def batch_rodrigues(rot_vecs, dtype=torch.float32):
    """ Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.Tensor Nx3
            array of N axis-angle vectors
        dtype
        Returns
        -------
        R: torch.Tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    k = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * k + (1 - cos) * torch.bmm(k, k)
    return rot_mat


def transform_mat(r, t):
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([func.pad(r, [0, 0, 0, 1]),
                      func.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.Tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.Tensor BxNx3
        Locations of joints
    parents : torch.Tensor BxN
        The kinematic tree of each object

    Returns
    -------
    posed_joints : torch.Tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.Tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # transforms_mat = transform_mat(
    #     rot_mats.view(-1, 3, 3),
    #     rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)
    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = func.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - func.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def lbs(betas, pose, v_template, shapedirs, posedirs, j_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32, detach_pose_correctives=True):
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.Tensor BxNB
            The Tensor of shape parameters
        pose : torch.Tensor Bx(j + 1) * 3
            The pose parameters in axis-angle format
        v_template: torch.Tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.Tensor 1xNB
            The Tensor of PCA shape displacements
        posedirs : torch.Tensor Px(V * 3)
            The pose PCA coefficients
        j_regressor : torch.Tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.Tensor j
            The array that describes the kinematic tree for the model
        lbs_weights: torch.Tensor N x V x (j + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose Tensor to rotation
            matrices. The default value is True. If False, then the pose Tensor
            should already contain rotation matrices and have a size of
            Bx(j + 1)x9
        dtype: torch.dtype, optional
        detach_pose_correctives

        Returns
        -------
        verts: torch.Tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.Tensor BxJx3
            The joints of the model
    """

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    j = vertices2joints(j_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x j x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    if detach_pose_correctives:
        pose_offsets = pose_offsets.detach()

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    j_transformed, a = batch_rigid_transform(rot_mats, j, parents)

    # 5. Do skinning:
    # w is N x V x (j + 1)
    w = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (j + 1)) x (N x (j + 1) x 16)
    num_joints = j_regressor.shape[0]
    t = torch.matmul(w, a.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(t, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, j_transformed


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    """ Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.Tensor BxVx3, dtype = torch.float32
            The Tensor of input vertices
        faces: torch.Tensor Fx3, dtype = torch.Long
            The faces of the mesh
        lmk_faces_idx: torch.Tensor L, dtype = torch.Long
            The Tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.Tensor Lx3, dtype = torch.float32
            The Tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.Tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    """
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks
