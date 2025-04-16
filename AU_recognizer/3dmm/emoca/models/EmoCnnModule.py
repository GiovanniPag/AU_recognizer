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
import torchvision.models.vgg as vgg
from torch.nn import Linear

from ..layers.losses.FRNet import resnet50, load_state_dict
from .EmotionRecognitionModuleBase import EmotionRecognitionBaseModule


class EmoCnnModule(EmotionRecognitionBaseModule):
    """
    Emotion Recognition module which uses a conv net as its backbone. Currently, Resnet-50 and VGG are supported.
    ResNet-50 based emotion recognition trained on AffectNet is the network used for self-supervising emotion in EMOCA.
    """

    def __init__(self, config):
        super().__init__(config)
        self.n_expression = config.data.n_expression if 'n_expression' in config.data.keys() else 9
        self.num_outputs = 0
        if self.config.model.predict_expression:
            self.num_outputs += self.n_expression
            self.num_classes = self.n_expression
        if self.config.model.predict_valence:
            self.num_outputs += 1
        if self.config.model.predict_arousal:
            self.num_outputs += 1
        if config.model.backbone == "resnet50":
            self.backbone = resnet50(num_classes=8631, include_top=False)
            if config.model.load_pretrained:
                load_state_dict(self.backbone, config.model.pretrained_weights)
            self.last_feature_size = 2048
            self.linear = Linear(self.last_feature_size, self.num_outputs)
            # 2048 is the output of  the resnet50 backbone without the MLP "top"
        elif config.model.backbone[:3] == "vgg":
            vgg_constructor = getattr(vgg, config.model.backbone)
            self.backbone = vgg_constructor(pretrained=bool(config.model.load_pretrained), progress=True)
            self.last_feature_size = 1000
            self.linear = Linear(self.last_feature_size, self.num_outputs)
            # 1000 is the number of imagenet classes so the dimension of the output of the vgg backbone
        else:
            raise ValueError(f"Invalid backbone: '{self.config.model.backbone}'")

    def get_last_feature_size(self):
        return self.last_feature_size

    def _forward(self, images):
        output = self.backbone(images)
        emo_feat_2 = output
        output = self.linear(output.view(output.shape[0], -1))
        out_idx = 0
        if self.predicts_expression():
            expr_classification = output[:, out_idx:(out_idx + self.n_expression)]
            if self.exp_activation is not None:
                expr_classification = self.exp_activation(expr_classification, dim=1)
            out_idx += self.n_expression
        else:
            expr_classification = None
        if self.predicts_valence():
            valence = output[:, out_idx:(out_idx + 1)]
            if self.v_activation is not None:
                valence = self.v_activation(valence)
            out_idx += 1
        else:
            valence = None
        if self.predicts_arousal():
            arousal = output[:, out_idx:(out_idx + 1)]
            if self.a_activation is not None:
                arousal = self.a_activation(arousal)
            out_idx += 1
        else:
            arousal = None
        assert out_idx == output.shape[1]
        values = {"emo_feat_2": emo_feat_2, "valence": valence, "arousal": arousal,
                  "expr_classification": expr_classification}
        return values

    def forward(self, batch):
        images = batch['image']
        if len(images.shape) != 5 and len(images.shape) != 4:
            raise RuntimeError("Invalid image batch dimensions.")
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        emotion = self._forward(images)
        valence = emotion['valence']
        arousal = emotion['arousal']
        values = {}
        if self.predicts_valence():
            values['valence'] = valence.view(-1, 1)
        if self.predicts_arousal():
            values['arousal'] = arousal.view(-1, 1)
        values['expr_classification'] = emotion['expr_classification']
        # WARNING: HACK
        if 'n_expression' not in self.config.data:
            if self.n_expression == 8:
                raise NotImplementedError("This here should not be called")
        return values

    def get_trainable_parameters(self):
        return list(self.backbone.parameters())

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        return None
