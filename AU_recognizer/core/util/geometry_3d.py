import math
import numpy as np
import numba
from AU_recognizer.core.util import time_me


@numba.njit(nogil=True, cache=True)
def max_3d_array(arr: np.ndarray, axis: int) -> float:
    """
    Find the maximum in a multidimensional array, in the provided axis
    """
    max_ = -np.inf
    for i in arr:
        if i[axis] >= max_:
            max_ = i[axis]
    return max_


@numba.njit(nogil=True, cache=True)
def min_3d_array(arr: np.ndarray, axis: int) -> float:
    """
    Find the minimum in a multidimensional array, in the provided axis
    """
    min_ = np.inf
    for i in arr:
        if i[axis] <= min_:
            min_ = i[axis]
    return min_


@numba.njit(nogil=True, cache=True, fastmath=True)
def normalize_3d_array(arr: np.ndarray,
                       nor_range: 'tuple(float, float)' = (-1, 1),
                       axis: int = 2
                       ) -> np.ndarray:
    """
    @brief: Normalize an array values within a range based on a specified axis
    @param arr: The array to be normalized
    @param nor_range: Normalized values range (min, max)
    @param axis: the axis to normalize based on
    @return arr: The normalized array
    """
    mnx = min_3d_array(arr, 0)
    mxx = max_3d_array(arr, 0)
    mnz = min_3d_array(arr, 2)
    mxz = max_3d_array(arr, 2)
    mny = min_3d_array(arr, 1)
    mxy = max_3d_array(arr, 1)

    if axis == 0:
        diff = mxx - mnx
    elif axis == 1:
        diff = mxy - mny
    else:
        diff = mxz - mnz

    for pt in arr:
        pt[0] = (((pt[0] - mnx) * (nor_range[1] - nor_range[0])) / diff) + nor_range[0]
        pt[1] = (((pt[1] - mny) * (nor_range[1] - nor_range[0])) / diff) + nor_range[0]
        pt[2] = (((pt[2] - mnz) * (nor_range[1] - nor_range[0])) / diff) + nor_range[0]
    return arr


@numba.njit(nogil=True, cache=True, fastmath=True)
def transform_point(point: np.ndarray,
                    orientation_quat: np.ndarray,
                    zoom: float,
                    obj_position: 'list[int, int]',
                    obj_scale: int
                    ) -> 'tuple(int, int)':
    """
    @brief: Rotate the point in
    3axis according to the provided rotation matrices
    @param point: 3D point
    @param orientation_quat: Orientation quaternion
    @param zoom: Zoom value
    @param obj_position: Object position within the screen
    @param obj_scale: Object scale
    @return transformed point: 2D transformed projection of the 3D point
    """
    # Rotate point on the Y, X, and Z axis respectively
    # Extract components from orientation quaternion
    p_quat = np.array([0.0, point[0], point[1], point[2]])
    # Assuming orientation_quat is normalized
    q_inverse = np.array([orientation_quat[0], -orientation_quat[1], -orientation_quat[2], -orientation_quat[3]])
    rotated_p_quat = quaternion_multiply(quaternion_multiply(orientation_quat, p_quat), q_inverse)

    # Extract rotated point
    rotated_2d = rotated_p_quat[1:].reshape((3, 1))  # Ignore the scalar part
    # Project 3D point on 2D plane
    z = 0.5 / (zoom - rotated_2d[2][0])
    projection_matrix = np.array(((z, 0, 0), (0, z, 0)))
    projected_2d = matmul(projection_matrix, rotated_2d)

    x = int(projected_2d[0][0] * obj_scale) + obj_position[0]
    # The (-) sign in the Y is because the canvas' Y axis starts from Top to Bottom,
    # so without the (-) sign, our objects would be presented upside down
    y = -int(projected_2d[1][0] * obj_scale) + obj_position[1]

    return x, y


@time_me
@numba.njit(nogil=True, cache=True, fastmath=True)
def transform_object(vertices, orientation_quat, zoom, obj_position, object_scale) -> 'list(list(int, int))':
    """Return the points of the object transformed according to the current pose"""
    projected_points = []
    for pt in vertices:
        x, y = transform_point(pt, orientation_quat, zoom, obj_position, object_scale)
        projected_points.append([x, y])
    return projected_points


@numba.njit(nogil=True, cache=True, fastmath=True)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    @brief: Matrix multiplication from scratch
    @param A: First matrix
    @param B: Second matrix
    @return C: Product of A and B
    """
    rows, cols = A.shape[0], B.shape[1]
    C = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            for k in range(rows):
                C[i, j] += A[i, k] * B[k, j]
    return C


@numba.njit(nogil=True, cache=True, fastmath=True)
def axis_angle_to_quaternion(axis, angle):
    """
    Convert an axis-angle representation to a quaternion.
    :param axis: A 3-element array-like object representing the rotation axis.
    :param angle: The rotation angle in radians.
    :return: A 4-element array representing the quaternion (x, y, z, w).
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    s = np.sin(angle / 2)
    c = np.cos(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


@numba.njit(nogil=True, cache=True, fastmath=True)
def quaternion_to_euler(q):
    """Convert a quaternion to Euler angles (x, y, z)"""
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


@numba.njit(nogil=True, cache=True, fastmath=True)
def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    :param q1: A 4-element array-like object representing the first quaternion (x, y, z, w).
    :param q2: A 4-element array-like object representing the second quaternion (x, y, z, w).
    :return: A 4-element array representing the resulting quaternion.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w])


@numba.njit(nogil=True, cache=True, fastmath=True)
def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ])

@numba.njit(nogil=True, cache=True)
def is_face_visible_2D(face):
    # Extract vertices
    v0 = np.array(face[0])
    v1 = np.array(face[1])
    v2 = np.array(face[2])  # Assuming face is a triangle in 2D
    # Calculate normal vector (perpendicular vector in 2D)
    normal = np.cross(v1 - v0, v2 - v0)  # Cross product for 2D (z-component)
    # Assuming clockwise is front-facing
    return normal < 0


class Geometry3D:
    OBJECT_SCALE = 2000  # Maybe make this dynamic depending on the object size

    def __init__(self, canvas_width: int, canvas_height: int) -> None:
        self._original_v = None
        self._obj_position = np.array((canvas_width // 2, canvas_height // 2))
        self._zoom = 10.0
        self._orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        self._faces = None
        self._vertices = None

    def upload_object(self, vert: np.ndarray, faces: list) -> None:
        """Uploads the vertices and faces to manipulate"""
        self._original_v = np.copy(vert)
        self._vertices = normalize_3d_array(vert, axis=0)
        self._faces = faces

    def update_position(self, x: int, y: int) -> None:
        """Update x, y position of the object"""
        self._obj_position[0] += x
        self._obj_position[1] += y

    def transform_object(self) -> 'list(list(int, int))':
        """Return the points of the object transformed according to the current pose"""
        return transform_object(self._vertices, self._orientation_quat, self._zoom, self._obj_position,
                                self.OBJECT_SCALE)

    @property
    def faces(self) -> list:
        """Get the faces formed between the points"""
        return self._faces

    @property
    def zoom(self) -> float:
        """Get the current zoom value"""
        return self._zoom

    @property
    def orientation(self) -> 'tuple(float, float, float, float)':
        """Returns the object's current orientation quaternion"""
        return self._orientation_quat

    def set_zoom(self, zoom: float) -> None:
        """Set the new zoom value"""
        self._zoom = zoom

    def reset_rotation(self,
                       axis_angle: tuple = (1.0, 0.0, 0.0, 0.0)
                       ) -> None:
        """Reset the rotation to a specific position, if provided, else to identity"""
        self._orientation_quat = axis_angle

    def set_rotation_from_xyz(self, x_angle, y_angle, z_angle):
        qx = axis_angle_to_quaternion(np.array([1, 0, 0]), x_angle)
        qy = axis_angle_to_quaternion(np.array([0, 1, 0]), y_angle)
        qz = axis_angle_to_quaternion(np.array([0, 0, 1]), z_angle)
        self._orientation_quat = quaternion_multiply(quaternion_multiply(qz, qy), qx)

    def update_scale(self, width, height):
        max_object_dimension = (
                max_3d_array(self._original_v, axis=(0 if width < height else 1)) - min_3d_array(self._original_v,
                                                                                                 axis=(
                                                                                                     0 if width < height else 1)))
        self.OBJECT_SCALE = (width if width < height else height) / max_object_dimension
