"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
import numpy as np


def bbox2point(left, right, top, bottom, l_type='bbox'):
    """ bbox from detector and landmarks are different
    """
    if l_type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif l_type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    elif l_type == "mediapipe":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    else:
        raise NotImplementedError(f" bbox2point not implemented for {l_type} ")
    if isinstance(center_x, np.ndarray):
        center = np.stack([center_x, center_y], axis=1)
    else:
        center = np.array([center_x, center_y])
    return old_size, center
