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
import torch.nn.functional as func
import sys

from ..layers.losses.EmonetLoader import get_emonet
from .DECA import instantiate_deca
from .EmotionRecognitionModuleBase import EmotionRecognitionBaseModule
from .MLP import MLP
from ..utils.other import class_from_str


class EmoDECA(EmotionRecognitionBaseModule):
    """
    EmoDECA loads a pretrained DECA-based face reconstruction net and uses it to predict emotion
    """

    def __init__(self, config):
        super().__init__(config)
        deca_checkpoint = config.model.deca_checkpoint
        deca_stage = config.model.deca_stage
        config.model.deca_cfg.model.background_from_input = False
        deca_checkpoint_kwargs = {
            "model_params": config.model.deca_cfg.model,
            "learning_params": config.model.deca_cfg.learning,
            "inout_params": config.model.deca_cfg.inout,
            "stage_name": "testing",
        }
        # instantiate the face net
        self.deca = instantiate_deca(config.model.deca_cfg, deca_stage, "test", deca_checkpoint, deca_checkpoint_kwargs)
        self.deca.inout_params.full_run_dir = config.inout.full_run_dir
        self._setup_deca(False)
        # which latent codes are being used
        in_size = 0
        if self.config.model.use_identity:
            in_size += config.model.deca_cfg.model.n_shape
        if self.config.model.use_expression:
            in_size += config.model.deca_cfg.model.n_exp
        if self.config.model.use_global_pose:
            in_size += 3
        if self.config.model.use_jaw_pose:
            in_size += 3
        if self.config.model.use_detail_code:
            in_size += config.model.deca_cfg.model.n_detail
        if 'mlp_dim' in self.config.model.keys():
            dimension = self.config.model.mlp_dim
        else:
            dimension = in_size
        hidden_layer_sizes = config.model.num_mlp_layers * [dimension]
        out_size = 0
        if self.predicts_expression():
            self.num_classes = self.config.data.n_expression if 'n_expression' in self.config.data.keys() else 9
            out_size += self.num_classes
        if self.predicts_valence():
            out_size += 1
        if self.predicts_arousal():
            out_size += 1
        if "use_mlp" not in self.config.model.keys() or self.config.model.use_mlp:
            if 'mlp_norm_layer' in self.config.model.keys():
                batch_norm = class_from_str(self.config.model.mlp_norm_layer, sys.modules[__name__])
            else:
                batch_norm = None
            self.mlp = MLP(in_size, out_size, hidden_layer_sizes, batch_norm=batch_norm)
        else:
            self.mlp = None
        if "use_emonet" in self.config.model.keys() and self.config.model.use_emonet:
            self.emonet = get_emonet(load_pretrained=config.model.load_pretrained_emonet)
            if not config.model.load_pretrained_emonet:
                self.emonet.n_expression = self.num_classes  # we use all affectnet classes (included none) for now
                self.emonet.create_Emo()  # reinitialize
        else:
            self.emonet = None

    def get_trainable_parameters(self):
        trainable_params = []
        if self.config.model.finetune_deca:  # false
            trainable_params += self.deca.get_trainable_parameters()
        if self.mlp is not None:
            trainable_params += list(self.mlp.parameters())
        if self.emonet is not None:
            trainable_params += list(self.emonet.parameters())
        return trainable_params

    def _setup_deca(self, train: bool):
        if self.config.model.finetune_deca:  # false
            self.deca.train(train)
            self.deca.requires_grad_(True)
        else:
            self.deca.train(False)
            self.deca.requires_grad_(False)

    def train(self, mode=True):
        self._setup_deca(mode)
        if self.mlp is not None:
            self.mlp.train(mode)
        if self.emonet is not None:
            self.emonet.train(mode)

    def emonet_out(self, images):
        images = func.interpolate(images, (256, 256), mode='bilinear')
        return self.emonet(images, intermediate_features=False)

    def forward_emonet(self, values, values_decoded, mode):
        if mode == 'detail':
            image_name = 'predicted_detailed_image'
        elif mode == 'coarse':
            image_name = 'predicted_images'
        else:
            raise ValueError(f"Invalid image mode '{mode}'")
        emotion = self.emonet_out(values_decoded[image_name])
        if self.v_activation is not None:
            emotion['valence'] = self.v_activation(emotion['valence'])
        if self.a_activation is not None:
            emotion['arousal'] = self.a_activation(emotion['arousal'])
        if self.exp_activation is not None:
            emotion['expression'] = self.exp_activation(emotion['expression'])
        values[f"emonet_{mode}_valence"] = emotion['valence'].view(-1, 1)
        values[f"emonet_{mode}_arousal"] = emotion['arousal'].view(-1, 1)
        values[f"emonet_{mode}_expr_classification"] = emotion['expression']
        return values

    def forward(self, batch):
        values = self.deca.encode(batch, training=False)
        shapecode = values['shapecode']
        expcode = values['expcode']
        posecode = values['posecode']
        jaw_pose = posecode[:, 3:]
        if self.mlp is not None:
            input_list = []
            if self.config.model.use_identity:
                input_list += [shapecode]
            if self.config.model.use_expression:
                input_list += [expcode]
            if self.config.model.use_jaw_pose:
                input_list += [jaw_pose]
            input_ = torch.cat(input_list, dim=1)
            output = self.mlp(input_)
            out_idx = 0
            if self.predicts_expression():
                expr_classification = output[:, out_idx:(out_idx + self.num_classes)]
                if self.exp_activation is not None:
                    expr_classification = self.exp_activation(output[:, out_idx:(out_idx + self.num_classes)], dim=1)
                out_idx += self.num_classes
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
                    arousal = self.a_activation(output[:, out_idx:(out_idx + 1)])
                out_idx += 1
            else:
                arousal = None
            values["valence"] = valence
            values["arousal"] = arousal
            values["expr_classification"] = expr_classification
        return values

    def _compute_loss(self,
                      pred, gt,
                      class_weight,
                      training=True,
                      **kwargs):
        if self.mlp is not None:
            losses_mlp, metrics_mlp = super()._compute_loss(pred, gt, class_weight, training, **kwargs)
        else:
            losses_mlp, metrics_mlp = {}, {}
        if self.emonet is not None:
            if self.config.model.use_coarse_image_emonet:
                losses_emonet_c, metrics_emonet_c = super()._compute_loss(pred, gt, class_weight, training,
                                                                          pred_prefix="emonet_coarse_", **kwargs)
            else:
                losses_emonet_c, metrics_emonet_c = {}, {}
            if self.config.model.use_detail_image_emonet:
                losses_emonet_d, metrics_emonet_d = super()._compute_loss(pred, gt, class_weight, training,
                                                                          pred_prefix="emonet_detail_", **kwargs)
            else:
                losses_emonet_d, metrics_emonet_d = {}, {}
            losses_emonet = {**losses_emonet_c, **losses_emonet_d}
            metrics_emonet = {**metrics_emonet_c, **metrics_emonet_d}
        else:
            losses_emonet, metrics_emonet = {}, {}
        losses = {**losses_emonet, **losses_mlp}
        metrics = {**metrics_emonet, **metrics_mlp}
        return losses, metrics

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        return None
