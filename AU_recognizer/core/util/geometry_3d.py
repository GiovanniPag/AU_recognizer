import math
import numpy as np
import numba


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
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])


# Function for perspective projection matrix
@numba.njit(nogil=True, cache=True, fastmath=True)
def perspective(fov, aspect, near, far):
    tan_half_fov = np.tan(np.radians(fov) / 2.0)
    return np.array([
        [1.0 / (aspect * tan_half_fov), 0, 0, 0],
        [0, 1.0 / tan_half_fov, 0, 0],
        [0, 0, -(far + near) / (far - near), -1],
        [0, 0, -(2.0 * far * near) / (far - near), 0]
    ])


# Function to create the view matrix (lookAt)
@numba.njit(nogil=True, cache=True, fastmath=True)
def look_at(eye, target, up):
    z = np.linalg.norm(eye - target)  # Eye-to-target direction
    x = np.cross(up, z)
    y = np.cross(z, x)
    return np.array([
        [x[0], y[0], z[0], eye[0]],
        [x[1], y[1], z[1], eye[1]],
        [x[2], y[2], z[2], eye[2]],
        [0, 0, 0, 1]
    ])

