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
from enum import Enum


class AffectNetExpressions(Enum):
    Neutral = 0
    Happy = 1
    Sad = 2
    Surprise = 3
    Fear = 4
    Disgust = 5
    Anger = 6
    Contempt = 7
    None_ = 8
    Uncertain = 9
    Occluded = 10
    xxx = 11

    @staticmethod
    def from_str(string: str):
        string = string[0].upper() + string[1:]
        return AffectNetExpressions[string]
