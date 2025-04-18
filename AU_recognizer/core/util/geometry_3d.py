import numba
import numpy as np


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
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])


def perspective(fov, aspect, near, far):
    tan_half_fov = np.tan(np.radians(fov) / 2)
    m = np.zeros((4, 4))

    m[0, 0] = 1 / (aspect * tan_half_fov)
    m[1, 1] = 1 / tan_half_fov
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -(2 * far * near) / (far - near)
    m[3, 2] = -1

    return m


# noinspection PyUnreachableCode
def look_at(eye, target, up):
    # Normalize the forward, right, and up vectors
    f = (target - eye)
    f /= np.linalg.norm(f) if np.linalg.norm(f) != 0 else 1
    r = np.cross(up, f)
    r /= np.linalg.norm(r) if np.linalg.norm(r) != 0 else 1
    u = np.cross(f, r)
    u /= np.linalg.norm(u) if np.linalg.norm(u) != 0 else 1
    # Create the lookAt matrix
    m = np.eye(4)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(r, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m
