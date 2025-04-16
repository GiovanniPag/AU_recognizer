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
from pathlib import Path

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import rescale, estimate_transform, warp

from ..utils.FaceDetector import FAN
from ..datasets.ImageDatasetHelpers import bbox2point


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0, max_detection=None):
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        else:
            testpath = Path(testpath)
            if testpath.is_dir():
                self.imagepath_list = list(testpath.glob("*.jpg")) + list(testpath.glob("*.png")) + list(
                    testpath.glob("*.bmp"))
            elif testpath.is_file() and testpath.suffix in {'.jpg', '.png', '.bmp'}:
                self.imagepath_list = [testpath]
            else:
                print(f'please check the test path: {testpath}')
                exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.max_detection = max_detection
        if face_detector == 'fan':
            self.face_detector = FAN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = str(self.imagepath_list[index])
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if self.scaling_factor != 1.:
            image = rescale(image, (self.scaling_factor, self.scaling_factor, 1)) * 255.

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = imagepath.replace('.jpg', '.mat').replace('.png', '.mat')
            kpt_txtpath = imagepath.replace('.jpg', '.txt').replace('.png', '.txt')
            if Path(kpt_matpath).exists():
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, l_type='kpt68')
            elif Path(kpt_txtpath).exists():
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, l_type='kpt68')
            else:
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 1:
                    print('no face detected! run original image')
                    left = 0
                    right = h - 1
                    top = 0
                    bottom = w - 1
                    old_size, center = bbox2point(left, right, top, bottom, l_type=bbox_type)
                else:
                    if self.max_detection is None:
                        bbox = bbox[0]
                        left = bbox[0]
                        right = bbox[2]
                        top = bbox[1]
                        bottom = bbox[3]
                        old_size, center = bbox2point(left, right, top, bottom, l_type=bbox_type)
                    else:
                        old_size, center = [], []
                        num_det = min(self.max_detection, len(bbox))
                        for bbi in range(num_det):
                            bb = bbox[0]
                            left = bb[0]
                            right = bb[2]
                            top = bb[1]
                            bottom = bb[3]
                            osz, c = bbox2point(left, right, top, bottom, l_type=bbox_type)
                        old_size += [osz]
                        center += [c]

            if isinstance(old_size, list):
                size = []
                src_pts = []
                for i in range(len(old_size)):
                    size += [int(old_size[i] * self.scale)]
                    src_pts += [np.array(
                        [[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2],
                         [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
                         [center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]
            else:
                size = int(old_size * self.scale)
                src_pts = np.array(
                    [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                     [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        image = image / 255.
        if not isinstance(src_pts, list):
            dst_pts = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, dst_pts)
            dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
            dst_image = dst_image.transpose(2, 0, 1)
            return {'image': torch.tensor(dst_image).float(),
                    'image_name': imagename,
                    'image_path': imagepath,
                    }
        else:
            dst_pts = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            dst_images = []
            for i in range(len(src_pts)):
                tform = estimate_transform('similarity', src_pts[i], dst_pts)
                dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
                dst_image = dst_image.transpose(2, 0, 1)
                dst_images += [dst_image]
            dst_images = np.stack(dst_images, axis=0)
            imagenames = [imagename + f"{j:02d}" for j in range(dst_images.shape[0])]
            imagepaths = [imagepath] * dst_images.shape[0]
            return {'image': torch.tensor(dst_images).float(),
                    'image_name': imagenames,
                    'image_path': imagepaths,
                    }
