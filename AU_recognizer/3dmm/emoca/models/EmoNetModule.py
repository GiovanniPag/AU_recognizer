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

import torch
import numpy as np
import torch.nn.functional as func

from ..datasets.AffWild2Dataset import Expression7
from ..datasets.AffectNetDataModule import AffectNetExpressions
from ..layers.losses.EmonetLoader import get_emonet
from .EmotionRecognitionModuleBase import EmotionRecognitionBaseModule


class EmoNetModule(EmotionRecognitionBaseModule):
    """
    Emotion analysis using the EmoNet architecture.
    https://github.com/face-analysis/emonet
    """

    def __init__(self, config):
        super().__init__(config)
        self.emonet = get_emonet(load_pretrained=config.model.load_pretrained_emonet)
        if not config.model.load_pretrained_emonet:
            n_expression = config.data.n_expression if 'n_expression' in config.data.keys() else 9
            self.emonet.n_expression = n_expression  # we use all affectnet classes (included none) for now
            self.n_expression = n_expression  # we use all affectnet classes (included none) for now
            self.emonet.create_Emo()  # reinitialize
        else:
            self.n_expression = 8
        self.num_classes = self.n_expression
        self.size = (256, 256)  # predefined input image size

    def emonet_out(self, images, intermediate_features=False):
        images = func.interpolate(images, self.size, mode='bilinear')
        return self.emonet(images, intermediate_features=intermediate_features)

    def _forward(self, images):
        if len(images.shape) != 5 and len(images.shape) != 4:
            raise RuntimeError("Invalid image batch dimensions.")
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        emotion = self.emonet_out(images, intermediate_features=True)
        valence = emotion['valence']
        arousal = emotion['arousal']
        if self.exp_activation is not None:
            expression = self.exp_activation(emotion['expression'], dim=1)
        values = {'valence': valence.view(-1, 1), 'arousal': arousal.view(-1, 1), 'expr_classification': expression}
        # WARNING: HACK
        if 'n_expression' not in self.config.data:
            if self.n_expression == 8:
                values['expr_classification'] = torch.cat([
                    values['expr_classification'],
                    torch.zeros_like(values['expr_classification'][:, 0:1]) + 2 * values['expr_classification'].min()],
                    dim=1)
        return values

    def forward(self, batch):
        images = batch['image']
        return self._forward(images)

    def get_trainable_parameters(self):
        return list(self.emonet.parameters())

    @staticmethod
    def _vae_2_str(valence=None, arousal=None, affnet_expr=None, expr7=None, prefix=""):
        caption = ""
        if len(prefix) > 0:
            prefix += "_"
        if valence is not None and not np.isnan(valence).any():
            caption += prefix + "valence= %.03f\n" % valence
        if arousal is not None and not np.isnan(arousal).any():
            caption += prefix + "arousal= %.03f\n" % arousal
        if affnet_expr is not None and not np.isnan(affnet_expr).any():
            caption += prefix + "expression= %s \n" % AffectNetExpressions(affnet_expr).name
        if expr7 is not None and not np.isnan(expr7).any():
            caption += prefix + "expression= %s \n" % Expression7(expr7).name
        return caption

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        return None
