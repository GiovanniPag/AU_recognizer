import sys
from enum import Enum
from pathlib import Path

import cv2
import torch
import torchvision
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import LightningModule
from skimage.io import imread
import torch.nn.functional as func
import numpy as np

from ..layers.losses import DecaLosses as lossFunc
from ..layers.losses.EmoNetLoss import create_emo_loss
from ..layers.losses.VGGLoss import VGG19Loss
from .DecaDecoder import Generator
from .DecaEncoder import ResnetEncoder
from .DecaFLAME import FLAME, FLAME_mediapipe, FLAMETex
from .Renderer import SRenderY
from ..utils.DecaUtils import load_local_mask, copy_state_dict, batch_orth_proj, \
    vertex_normals, tensor_vis_landmarks
from ..utils.other import class_from_str

torch.backends.cudnn.benchmark = True


class DecaMode(Enum):
    COARSE = 1  # when switched on, only coarse part of DECA-based networks is used
    DETAIL = 2  # when switched on, only coarse and detail part of DECA-based networks is used


# noinspection PyUnboundLocalVariable
class DecaModule(LightningModule):
    """
    DecaModule is a PL module that implements DECA-inspired face reconstruction networks.
    """

    def __init__(self, model_params, learning_params, inout_params, stage_name=""):
        """
        :param model_params: a DictConfig of parameters about the model itself :param learning_params: a DictConfig
        of parameters corresponding to the learning process (such as optimizer, lr and others) :param inout_params: a
        DictConfig of parameters about input and output (where checkpoints and visualizations are saved)
        """
        super().__init__()
        self.learning_params = learning_params
        self.inout_params = inout_params
        # detail conditioning - what is given as the conditioning input to the detail generator in detail stage training
        if 'detail_conditioning' not in model_params.keys():
            # jaw, expression and detail code by default
            self.detail_conditioning = ['jawpose', 'expression', 'detail']
            OmegaConf.set_struct(model_params, True)
            with open_dict(model_params):
                model_params.detail_conditioning = self.detail_conditioning
        else:
            self.detail_conditioning = model_params.detail_conditioning
        supported_conditioning_keys = ['identity', 'jawpose', 'expression', 'detail']

        for c in self.detail_conditioning:
            if c not in supported_conditioning_keys:
                raise ValueError(
                    f"Conditioning on '{c}' is not supported. Supported conditionings: {supported_conditioning_keys}")
        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])
        # instantiate the network
        self.deca = deca_class(config=model_params)
        self.mode = DecaMode[str(model_params.mode).upper()]
        self.stage_name = stage_name
        if self.stage_name is None:
            self.stage_name = ""
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        # initialize the emotion perceptual loss (used for EMOCA supervision)
        self.emonet_loss = None
        self._init_emotion_loss()

    def _init_emotion_loss(self):
        """
        Initialize the emotion perceptual loss (used for EMOCA supervision)
        """
        if 'emonet_weight' in self.deca.config.keys() and bool(self.deca.config.emonet_model_path):
            if self.emonet_loss is not None:
                if self.emonet_loss.is_trainable():
                    print("The old emonet loss is trainable and will not be overrided or replaced.")
                    return
                else:
                    print("The old emonet loss is not trainable. It will be replaced.")
            if 'emonet_model_path' in self.deca.config.keys():
                emonet_model_path = self.deca.config.emonet_model_path
            else:
                emonet_model_path = None
            emo_feat_loss = self.deca.config.emo_feat_loss if 'emo_feat_loss' in self.deca.config.keys() else None
            old_emonet_loss = self.emonet_loss
            self.emonet_loss = create_emo_loss(self.device, emoloss=emonet_model_path, trainable=False,
                                               dual=False,
                                               normalize_features=None,
                                               emo_feat_loss=emo_feat_loss)
            if old_emonet_loss is not None and type(old_emonet_loss) is not self.emonet_loss:
                print(f"The old emonet loss {old_emonet_loss.__class__.__name__} is replaced during reconfiguration by "
                      f"new emotion loss {self.emonet_loss.__class__.__name__}")
        else:
            self.emonet_loss = None

    def _encode_flame(self, images):
        if self.mode == DecaMode.COARSE or \
                (self.mode == DecaMode.DETAIL and self.deca.config.train_coarse):
            # forward pass with gradients (for coarse stage (used), or detail stage with coarse training (not used))
            parameters = self.deca.encode_flame(images)
        elif self.mode == DecaMode.DETAIL:
            # in detail stage, the coarse forward pass does not need gradients
            with torch.no_grad():
                parameters = self.deca.encode_flame(images)
        else:
            raise ValueError(f"Invalid EMOCA Mode {self.mode}")
        code_list, original_code = self.deca.decompose_code(parameters)
        return code_list, original_code

    @staticmethod
    def _unwrap_list(codelist):
        shapecode, texcode, expcode, posecode, cam, lightcode = codelist
        return shapecode, texcode, expcode, posecode, cam, lightcode

    @staticmethod
    def _unwrap_list_to_dict(codelist):
        shapecode, texcode, expcode, posecode, cam, lightcode = codelist
        return {'shape': shapecode, 'tex': texcode, 'exp': expcode, 'pose': posecode, 'cam': cam, 'light': lightcode}

    def encode(self, batch) -> dict:
        """
        Forward encoding pass of the model. Takes a batch of images and returns
        the corresponding latent codes for each image.
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size].
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'.
        For a testing pass, the images suffice.
        """
        codedict = {}
        images = batch['image']
        if len(images.shape) != 5 and len(images.shape) != 4:
            raise RuntimeError("Invalid image batch dimensions.")
        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        if 'landmark' in batch.keys():
            lmk = batch['landmark']
            lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        if 'landmark_mediapipe' in batch.keys():
            lmk_mp = batch['landmark_mediapipe']
            lmk_mp = lmk_mp.view(-1, lmk_mp.shape[-2], lmk_mp.shape[-1])
        else:
            lmk_mp = None
        if 'mask' in batch.keys():
            masks = batch['mask']
            masks = masks.view(-1, images.shape[-2], images.shape[-1])
        # 1) COARSE STAGE
        # forward pass of the coarse encoder
        code, original_code = self._encode_flame(images)
        shapecode, texcode, expcode, posecode, cam, lightcode = self._unwrap_list(code)
        if original_code is not None:
            original_code = self._unwrap_list_to_dict(original_code)
        # 2) DETAIL STAGE
        if self.mode == DecaMode.DETAIL:
            all_detailcode = self.deca.E_detail(images)
            # identity-based detail code
            detailcode = all_detailcode[:, :self.deca.n_detail]
            # detail emotion code is deprecated and will be empty
            detailemocode = all_detailcode[:, self.deca.n_detail:self.deca.n_detail]
        codedict['shapecode'] = shapecode
        codedict['texcode'] = texcode
        codedict['expcode'] = expcode
        codedict['posecode'] = posecode
        codedict['cam'] = cam
        codedict['lightcode'] = lightcode
        if self.mode == DecaMode.DETAIL:
            codedict['detailcode'] = detailcode
            codedict['detailemocode'] = detailemocode
        codedict['images'] = images
        if 'mask' in batch.keys():
            codedict['masks'] = masks
        if 'landmark' in batch.keys():
            codedict['lmk'] = lmk
        if lmk_mp is not None:
            codedict['lmk_mp'] = lmk_mp
        if original_code is not None:
            codedict['original_code'] = original_code
        return codedict

    def uses_texture(self):
        """
        Check if the model uses texture
        """
        return self.deca.uses_texture()

    def _create_conditioning_lists(self, codedict, condition_list):
        detail_conditioning_list = []
        if 'globalpose' in condition_list:
            detail_conditioning_list += [codedict["posecode"][:, :3]]
        if 'jawpose' in condition_list:
            detail_conditioning_list += [codedict["posecode"][:, 3:]]
        if 'identity' in condition_list:
            detail_conditioning_list += [codedict["shapecode"]]
        if 'expression' in condition_list:
            detail_conditioning_list += [codedict["expcode"]]
        if isinstance(self.deca.D_detail, Generator):
            # the detail codes might be excluded from conditioning based on the Generator architecture (for instance
            # for AdaIn Generator)
            if 'detail' in condition_list:
                detail_conditioning_list += [codedict["detailcode"]]
            if 'detailemo' in condition_list:
                detail_conditioning_list += [codedict["detailemocode"]]
        return detail_conditioning_list

    def decode(self, codedict, render=True) -> dict:
        """
        Forward decoding pass of the model. Takes the latent code predicted by the encoding stage and reconstructs and
        renders the shape.
        :param codedict: Batch dict of the predicted latent codes
        :param render: render
        """
        shapecode = codedict['shapecode']
        expcode = codedict['expcode']
        posecode = codedict['posecode']
        texcode = codedict['texcode']
        cam = codedict['cam']
        lightcode = codedict['lightcode']
        images = codedict['images']
        if 'masks' in codedict.keys():
            masks = codedict['masks']
        else:
            masks = None
        effective_batch_size = images.shape[0]
        # this is the current batch size after all training augmentations modifications
        # 1) Reconstruct the face mesh
        # FLAME - world space
        if not isinstance(self.deca.flame, FLAME_mediapipe):
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                              pose_params=posecode)
            landmarks2d_mediapipe = None
        else:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = self.deca.flame(shapecode, expcode, posecode)
        # world to camera
        trans_verts = batch_orth_proj(verts, cam)
        predicted_landmarks = batch_orth_proj(landmarks2d, cam)[:, :, :2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]
        if landmarks2d_mediapipe is not None:
            predicted_landmarks_mediapipe = batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]
            predicted_landmarks_mediapipe[:, :, 1:] = - predicted_landmarks_mediapipe[:, :, 1:]
        if self.uses_texture():
            albedo = self.deca.flametex(texcode)
        else:
            # if not using texture, default to gray
            albedo = torch.ones([effective_batch_size, 3, self.deca.config.uv_size, self.deca.config.uv_size],
                                device=images.device) * 0.5
        # 2) Render the coarse image
        if render:
            ops = self.deca.render(verts, trans_verts, albedo, lightcode)
            # mask
            mask_face_eye = func.grid_sample(self.deca.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                             ops['grid'].detach(),
                                             align_corners=False)
            # images
            predicted_images = ops['images']
            if isinstance(self.deca.config.useSeg, bool):
                if self.deca.config.useSeg:
                    segmentation_type = 'gt'
                else:
                    segmentation_type = 'rend'
            elif isinstance(self.deca.config.useSeg, str):
                segmentation_type = self.deca.config.useSeg
            else:
                raise RuntimeError(f"Invalid 'useSeg' type: '{type(self.deca.config.useSeg)}'")
            if segmentation_type not in ["gt", "rend", "intersection", "union"]:
                raise ValueError(f"Invalid segmentation type for masking '{segmentation_type}'")
            if masks is None:  # if mask not provided, the only mask available is the rendered one
                segmentation_type = 'rend'
            elif masks.shape[-1] != predicted_images.shape[-1] or masks.shape[-2] != predicted_images.shape[-2]:
                # resize masks if need be (this is only done if configuration was changed at some point after training)
                dims = masks.ndim == 3
                if dims:
                    masks = masks[:, None, :, :]
                masks = func.interpolate(masks, size=predicted_images.shape[-2:], mode='bilinear')
                if dims:
                    masks = masks[:, 0, ...]
            # resize images if need be (this is only done if configuration was changed at some point after training)
            if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                # special case only for inference time if the rendering image sizes have been changed
                images_resized = func.interpolate(images, size=predicted_images.shape[-2:], mode='bilinear')
            else:
                images_resized = images
            # what type of segmentation we use
            if segmentation_type == "gt":  # GT stands for external segmentation predicted by face parsing or similar
                masks = masks[:, None, :, :]
            elif segmentation_type == "rend":  # mask rendered as a silhouette of the face mesh
                masks = mask_face_eye * ops['alpha_images']
            elif segmentation_type == "intersection":  # intersection of the two above
                masks = masks[:, None, :, :] * mask_face_eye * ops['alpha_images']
            elif segmentation_type == "union":  # union of the first two options
                masks = torch.max(masks[:, None, :, :], mask_face_eye * ops['alpha_images'])
            else:
                raise RuntimeError(f"Invalid segmentation type for masking '{segmentation_type}'")
            if self.deca.config.background_from_input in [True, "input"]:
                if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                    # special case only for inference time if the rendering image sizes have been changed
                    predicted_images = (1. - masks) * images_resized + masks * predicted_images
                else:
                    predicted_images = (1. - masks) * images + masks * predicted_images
            elif self.deca.config.background_from_input in [False, "black"]:
                predicted_images = masks * predicted_images
            elif self.deca.config.background_from_input in ["none"]:
                predicted_images = predicted_images
            else:
                raise ValueError(f"Invalid type of background modification {self.deca.config.background_from_input}")
        # 3) Render the detail image
        if self.mode == DecaMode.DETAIL:
            # a) Create the detail conditioning lists
            detail_conditioning_list = self._create_conditioning_lists(codedict, self.detail_conditioning)
            final_detail_conditioning_list = detail_conditioning_list
            # b) Pass the detail code and the conditions through the detail generator to get displacement UV map
            if isinstance(self.deca.D_detail, Generator):
                uv_z = self.deca.D_detail(torch.cat(final_detail_conditioning_list, dim=1))
            else:
                raise ValueError(
                    f"This class of generarator is not supported: '{self.deca.D_detail.__class__.__name__}'")
            # render detail
            if render:
                detach_from_coarse_geometry = not self.deca.config.train_coarse
                uv_detail_normals, uv_coarse_vertices = (
                    self.deca.displacement2normal(uv_z, verts, ops['normals'], detach=detach_from_coarse_geometry))
                uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
                uv_texture = albedo.detach() * uv_shading
                # batch size X image_rows X image_cols X 2
                # you can query the grid for UV values of the face mesh at pixel locations
                grid = ops['grid']
                if detach_from_coarse_geometry:
                    # if the grid is detached, the gradient of the positions of UV-values
                    # in image space won't flow back to the geometry
                    grid = grid.detach()
                predicted_detailed_image = func.grid_sample(uv_texture, grid, align_corners=False)
                if self.deca.config.background_from_input in [True, "input"]:
                    if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                        # special case only for inference time if the rendering image sizes have been changed
                        predicted_detailed_image = (1. - masks) * images_resized + masks * predicted_detailed_image
                    else:
                        predicted_detailed_image = (1. - masks) * images + masks * predicted_detailed_image
                elif self.deca.config.background_from_input in [False, "black"]:
                    predicted_detailed_image = masks * predicted_detailed_image
                elif self.deca.config.background_from_input in ["none"]:
                    predicted_detailed_image = predicted_detailed_image
                else:
                    raise ValueError(
                        f"Invalid type of background modification {self.deca.config.background_from_input}")
                # --- extract texture
                uv_pverts = self.deca.render.world2uv(trans_verts).detach()
                uv_gt = func.grid_sample(torch.cat([images_resized, masks], dim=1),
                                         uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
                                         mode='bilinear', align_corners=True)
                uv_texture_gt = uv_gt[:, :3, :, :].detach()
                uv_mask_gt = uv_gt[:, 3:, :, :].detach()
                # self-occlusion
                normals = vertex_normals(trans_verts, self.deca.render.faces.expand(effective_batch_size, -1, -1))
                uv_pnorm = self.deca.render.world2uv(normals)
                uv_mask = (uv_pnorm[:, -1, :, :] < -0.05).float().detach()
                uv_mask = uv_mask[:, None, :, :]
                # combine masks
                uv_vis_mask = uv_mask_gt * uv_mask * self.deca.uv_face_eye_mask
        else:
            uv_detail_normals = None
            predicted_detailed_image = None
        # populate the value dict for metric computation/visualization
        if render:
            codedict['predicted_images'] = predicted_images
            codedict['predicted_detailed_image'] = predicted_detailed_image
            codedict['ops'] = ops
            codedict['normals'] = ops['normals']
            codedict['mask_face_eye'] = mask_face_eye
        codedict['verts'] = verts
        codedict['albedo'] = albedo
        codedict['landmarks2d'] = landmarks2d
        codedict['landmarks3d'] = landmarks3d
        codedict['predicted_landmarks'] = predicted_landmarks
        if landmarks2d_mediapipe is not None:
            codedict['predicted_landmarks_mediapipe'] = predicted_landmarks_mediapipe
        codedict['trans_verts'] = trans_verts
        codedict['masks'] = masks
        if self.mode == DecaMode.DETAIL:
            if render:
                codedict['uv_texture_gt'] = uv_texture_gt
                codedict['uv_texture'] = uv_texture
                codedict['uv_detail_normals'] = uv_detail_normals
                codedict['uv_shading'] = uv_shading
                codedict['uv_vis_mask'] = uv_vis_mask
                codedict['uv_mask'] = uv_mask
            codedict['uv_z'] = uv_z
            codedict['displacement_map'] = uv_z + self.deca.fixed_uv_dis[None, None, :, :]
        return codedict

    def visualization_checkpoint(self, verts, trans_verts, ops, uv_detail_normals, additional, batch_idx,
                                 stage, prefix, save=False):
        batch_size = verts.shape[0]
        visind = np.arange(batch_size)
        shape_images = self.deca.render.render_shape(verts, trans_verts)
        if uv_detail_normals is not None:
            detail_normal_images = func.grid_sample(uv_detail_normals.detach(), ops['grid'].detach(),
                                                    align_corners=False)
            shape_detail_images = self.deca.render.render_shape(verts, trans_verts,
                                                                detail_normal_images=detail_normal_images)
        else:
            shape_detail_images = None
        visdict = {}
        if 'images' in additional.keys():
            visdict['inputs'] = additional['images'][visind]
        if 'images' in additional.keys() and 'lmk' in additional.keys():
            visdict['landmarks_gt'] = tensor_vis_landmarks(additional['images'][visind], additional['lmk'][visind])
        if 'images' in additional.keys() and 'predicted_landmarks' in additional.keys():
            visdict['landmarks_predicted'] = tensor_vis_landmarks(additional['images'][visind],
                                                                  additional['predicted_landmarks'][visind])
        if 'predicted_images' in additional.keys():
            visdict['output_images_coarse'] = additional['predicted_images'][visind]
        if 'predicted_translated_image' in additional.keys() and additional['predicted_translated_image'] is not None:
            visdict['output_translated_images_coarse'] = additional['predicted_translated_image'][visind]
        visdict['geometry_coarse'] = shape_images[visind]
        if shape_detail_images is not None:
            visdict['geometry_detail'] = shape_detail_images[visind]
        if 'albedo_images' in additional.keys():
            visdict['albedo_images'] = additional['albedo_images'][visind]
        if 'masks' in additional.keys():
            visdict['mask'] = additional['masks'].repeat(1, 3, 1, 1)[visind]
        if 'albedo' in additional.keys():
            visdict['albedo'] = additional['albedo'][visind]
        if 'predicted_detailed_image' in additional.keys() and additional['predicted_detailed_image'] is not None:
            visdict['output_images_detail'] = additional['predicted_detailed_image'][visind]
        if ('predicted_detailed_translated_image' in additional.keys() and
                additional['predicted_detailed_translated_image'] is not None):
            visdict['output_translated_images_detail'] = additional['predicted_detailed_translated_image'][visind]
        if 'shape_detail_images' in additional.keys():
            visdict['shape_detail_images'] = additional['shape_detail_images'][visind]
        if 'uv_detail_normals' in additional.keys():
            visdict['uv_detail_normals'] = additional['uv_detail_normals'][visind] * 0.5 + 0.5
        if 'uv_texture_patch' in additional.keys():
            visdict['uv_texture_patch'] = additional['uv_texture_patch'][visind]
        if 'uv_texture_gt' in additional.keys():
            visdict['uv_texture_gt'] = additional['uv_texture_gt'][visind]
        if 'translated_uv_texture' in additional.keys() and additional['translated_uv_texture'] is not None:
            visdict['translated_uv_texture'] = additional['translated_uv_texture'][visind]
        if 'uv_vis_mask_patch' in additional.keys():
            visdict['uv_vis_mask_patch'] = additional['uv_vis_mask_patch'][visind]
        if save:
            savepath = (f'{self.inout_params.full_run_dir}/{prefix}_{stage}'
                        f'/combined/{self.current_epoch:04d}_{batch_idx:04d}.png')
            Path(savepath).parent.mkdir(exist_ok=True, parents=True)
            visualization_image = self.deca.visualize(visdict, savepath)
            return visdict, visualization_image[..., [2, 1, 0]]
        else:
            return visdict, None

    def visualize(self, visdict, savepath, catdim=1):
        return self.deca.visualize(visdict, savepath, catdim)

    def reconfigure(self, model_params, inout_params, learning_params, stage_name="", downgrade_ok=False, train=True):
        """
        Reconfigure the model. Usually used to switch between detail and coarse stages (which have separate configs)
        """
        if (self.mode == DecaMode.DETAIL and model_params.mode != DecaMode.DETAIL) and not downgrade_ok:
            raise RuntimeError("You're switching the EMOCA mode from DETAIL to COARSE. Is this really what you want?!")
        self.inout_params = inout_params
        self.learning_params = learning_params
        if self.deca.__class__.__name__ != model_params.deca_class:
            old_deca_class = self.deca.__class__.__name__
            state_dict = self.deca.state_dict()
            if 'deca_class' in model_params.keys():
                deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])
            else:
                deca_class = DECA
            self.deca = deca_class(config=model_params)
            diff = set(state_dict.keys()).difference(set(self.deca.state_dict().keys()))
            if len(diff) > 0:
                raise RuntimeError(f"Some values from old state dict will not be used. This is probably not what you "
                                   f"want because it most likely means that the pretrained model's weights won't be "
                                   f"used."
                                   f"Maybe you messed up backbone compatibility (i.e. SWIN vs ResNet?) {diff}")
            ret = self.deca.load_state_dict(state_dict, strict=False)
            if len(ret.unexpected_keys) > 0:
                raise print(f"Unexpected keys: {ret.unexpected_keys}")
            missing_modules = set([s.split(".")[0] for s in ret.missing_keys])
            print(f"Missing modules when upgrading from {old_deca_class} to {model_params.deca_class}:")
            print(missing_modules)
        else:
            self.deca.reconfigure(model_params)
        self._init_emotion_loss()
        self.stage_name = stage_name
        if self.stage_name is None:
            self.stage_name = ""
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        self.mode = DecaMode[str(model_params.mode).upper()]
        self.train(mode=train)
        print(f"EMOCA MODE RECONFIGURED TO: {self.mode}")
        if 'shape_contrain_type' in self.deca.config.keys() and str(
                self.deca.config.shape_constrain_type).lower() != 'none':
            shape_constraint = self.deca.config.shape_constrain_type
        else:
            shape_constraint = None
        if 'expression_constrain_type' in self.deca.config.keys() and str(
                self.deca.config.expression_constrain_type).lower() != 'none':
            expression_constraint = self.deca.config.expression_constrain_type
        else:
            expression_constraint = None
        if shape_constraint is not None and expression_constraint is not None:
            raise ValueError(
                "Both shape constraint and expression constraint are active. This is probably not what we want.")

    def train(self, mode: bool = True):
        self.deca.train(mode)
        if self.emonet_loss is not None:
            self.emonet_loss.eval()
        if self.deca.perceptual_loss is not None:
            self.deca.perceptual_loss.eval()
        if self.deca.id_loss is not None:
            self.deca.id_loss.eval()
        return self

    def get_trainable_parameters(self):
        trainable_params = []
        if self.mode == DecaMode.COARSE:
            trainable_params += self.deca.get_coarse_trainable_parameters()
        elif self.mode == DecaMode.DETAIL:
            trainable_params += self.deca.get_detail_trainable_parameters()
        else:
            raise ValueError(f"Invalid deca mode: {self.mode}")
        if self.emonet_loss is not None:
            trainable_params += self.emonet_loss.get_trainable_params()
        if self.deca.id_loss is not None:
            trainable_params += self.deca.id_loss.get_trainable_params()
        return trainable_params

    def cuda(self, device=None):
        super().cuda(device)
        return self


class DECA(torch.nn.Module):
    """
    The original DECA class which contains the encoders, FLAME decoder and the detail decoder.
    """

    def __init__(self, config):
        """
        :config corresponds to a model_params from DecaModule
        """
        super().__init__()

        # ID-MRF perceptual loss (kept here from the original DECA implementation)
        self.mode = None
        self.start_epoch = None
        self.start_iter = None
        self.n_cond = None
        self.n_detail = None
        self.n_param = None
        self.config = None
        self.perceptual_loss = None

        # Face Recognition loss
        self.id_loss = None

        # VGG feature loss
        self.vgg_loss = None

        self.reconfigure(config)
        self._reinitialize()

    def reconfigure(self, config):
        self.config = config

        self.n_param = config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        # identity-based detail code
        self.n_detail = config.n_detail

        # count the size of the condition vector
        if 'detail_conditioning' in self.config.keys():
            self.n_cond = 0
            if 'globalpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'jawpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'identity' in self.config.detail_conditioning:
                self.n_cond += config.n_shape
            if 'expression' in self.config.detail_conditioning:
                self.n_cond += config.n_exp
        else:
            self.n_cond = 3 + config.n_exp

        self.mode = DecaMode[str(config.mode).upper()]
        self._create_detail_generator()
        self._init_deep_losses()

    def _reinitialize(self):
        self._create_model()
        self._setup_renderer()
        self._init_deep_losses()
        self.face_attr_mask = load_local_mask(image_size=self.config.uv_size, mode='bbx')

    def _create_detail_generator(self):
        # backwards compatibility hack:
        if hasattr(self, 'D_detail'):
            if (
                    "detail_conditioning_type" not in self.config.keys() or
                    self.config.detail_conditioning_type == "concat") and isinstance(self.D_detail, Generator):
                return
            print("[WARNING]: We are reinitializing the detail generator!")
            del self.D_detail  # just to make sure we free the CUDA memory, probably not necessary
        if "detail_conditioning_type" not in self.config.keys() or str(
                self.config.detail_conditioning_type).lower() == "concat":
            # concatenates detail latent and conditioning (this one is used by DECA/EMOCA)
            print("Creating classic detail generator.")
            self.D_detail = Generator(latent_dim=self.n_detail + self.n_cond, out_channels=1,
                                      out_scale=0.01,
                                      sample_mode='bilinear')
        else:
            raise NotImplementedError(f"Detail conditioning invalid: '{self.config.detail_conditioning_type}'")

    def _get_num_shape_params(self):
        return self.config.n_shape

    def get_coarse_trainable_parameters(self):
        print("Add E_flame.parameters() to the optimizer")
        return list(self.E_flame.parameters())

    def get_detail_trainable_parameters(self):
        trainable_params = []
        if self.config.train_coarse:
            trainable_params += self.get_coarse_trainable_parameters()
            print("Add E_flame.parameters() to the optimizer")
        trainable_params += list(self.E_detail.parameters())
        print("Add E_detail.parameters() to the optimizer")
        trainable_params += list(self.D_detail.parameters())
        print("Add D_detail.parameters() to the optimizer")
        return trainable_params

    def _init_deep_losses(self):
        """
        Initialize networks for deep losses
        """
        # ideally these networks should be moved out the DECA class and into DecaModule,
        # but that would break backwards compatility with the original DECA and would not be able to load DECA's weights
        if 'mrfwr' not in self.config.keys() or self.config.mrfwr == 0:
            self.perceptual_loss = None
        else:
            if self.perceptual_loss is None:
                self.perceptual_loss = lossFunc.IDMRFLoss().eval()
                self.perceptual_loss.requires_grad_(False)  # move this to the constructor

        if 'idw' not in self.config.keys() or self.config.idw == 0:
            self.id_loss = None
        else:
            if self.id_loss is None:
                id_metric = self.config.id_metric if 'id_metric' in self.config.keys() else None
                id_trainable = self.config.id_trainable if 'id_trainable' in self.config.keys() else False
                self.id_loss_start_step = self.config.id_loss_start_step if 'id_loss_start_step' in self.config.keys() \
                    else 0
                self.id_loss = lossFunc.VGGFace2Loss(self.config.pretrained_vgg_face_path, id_metric, id_trainable)
                self.id_loss.freeze_nontrainable_layers()

        if 'vggw' not in self.config.keys() or self.config.vggw == 0:
            self.vgg_loss = None
        else:
            if self.vgg_loss is None:
                vgg_loss_batch_norm = 'vgg_loss_batch_norm' in self.config.keys() and self.config.vgg_loss_batch_norm
                self.vgg_loss = VGG19Loss(dict(zip(self.config.vgg_loss_layers, self.config.lambda_vgg_layers)),
                                          batch_norm=vgg_loss_batch_norm).eval()
                self.vgg_loss.requires_grad_(False)  # move this to the constructor

    def uses_texture(self):
        if 'use_texture' in self.config.keys():
            return self.config.use_texture
        return True  # true by default

    def _create_model(self):
        # 1) build coarse encoder
        e_flame_type = 'ResnetEncoder'
        if 'e_flame_type' in self.config.keys():
            e_flame_type = self.config.e_flame_type

        if e_flame_type == 'ResnetEncoder':
            self.E_flame = ResnetEncoder(outsize=self.n_param)
        else:
            raise ValueError(f"Invalid 'e_flame_type' = {e_flame_type}")

        import copy
        flame_cfg = copy.deepcopy(self.config)
        flame_cfg.n_shape = self._get_num_shape_params()
        if 'flame_mediapipe_lmk_embedding_path' not in flame_cfg.keys():
            self.flame = FLAME(flame_cfg)
        else:
            self.flame = FLAME_mediapipe(flame_cfg)

        if self.uses_texture():
            self.flametex = FLAMETex(self.config)
        else:
            self.flametex = None
        # 2) build detail encoder
        e_detail_type = 'ResnetEncoder'
        if 'e_detail_type' in self.config.keys():
            e_detail_type = self.config.e_detail_type
        if e_detail_type == 'ResnetEncoder':
            self.E_detail = ResnetEncoder(outsize=self.n_detail)
        else:
            raise ValueError(f"Invalid 'e_detail_type'={e_detail_type}")
        self._create_detail_generator()

    def encode_flame(self, images):
        return self.E_flame(images)

    def decompose_code(self, code):
        """
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        """
        code_list = []
        num_list = [self.config.n_shape, self.config.n_tex, self.config.n_exp, self.config.n_pose, self.config.n_cam,
                    self.config.n_light]
        start = 0
        for i in range(len(num_list)):
            code_list.append(code[:, start:start + num_list[i]])
            start = start + num_list[i]
        # shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        return code_list, None

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals, detach=True):
        """
        Converts the displacement uv map (uv_z) and coarse_verts to a normal map coarse_normals.
        """
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts)
        if detach:
            uv_coarse_vertices = uv_coarse_vertices.detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals)
        if detach:
            uv_coarse_normals = uv_coarse_normals.detach()
        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = (uv_coarse_vertices + uv_z * uv_coarse_normals +
                              self.fixed_uv_dis[None, None, :, :] * uv_coarse_normals)
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        return uv_detail_normals, uv_coarse_vertices

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.mode == DecaMode.COARSE:
                self.E_flame.train()
                self.E_detail.eval()
                self.D_detail.eval()
            elif self.mode == DecaMode.DETAIL:
                if self.config.train_coarse:
                    self.E_flame.train()
                else:
                    self.E_flame.eval()
                self.E_detail.train()
                self.D_detail.train()
            else:
                raise ValueError(f"Invalid mode '{self.mode}'")
        else:
            self.E_flame.eval()
            self.E_detail.eval()
            self.D_detail.eval()
        # these are set to eval no matter what, they're never being trained
        # (the FLAME shape and texture spaces are pretrained)
        self.flame.eval()
        if self.flametex is not None:
            self.flametex.eval()
        return self

    def _setup_renderer(self):
        self.render = SRenderY(self.config.image_size, obj_filename=self.config.topology_path,
                               uv_size=self.config.uv_size)
        # face mask for rendering details
        mask = imread(self.config.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = func.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        mask = imread(self.config.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = func.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)

        # displacement correct
        if Path(self.config.fixed_displacement_path).is_file():
            fixed_dis = np.load(self.config.fixed_displacement_path)
            fixed_uv_dis = torch.tensor(fixed_dis).float()
        else:
            fixed_uv_dis = torch.zeros([512, 512]).float()
            print("Warning: fixed_displacement_path not found, using zero displacement")
        self.register_buffer('fixed_uv_dis', fixed_uv_dis)

    def load_old_checkpoint(self):
        """
        Loads the DECA model weights from the original DECA implementation:
        https://github.com/YadiraF/DECA
        """
        if self.config.resume_training:
            model_path = self.config.pretrained_modelpath
            print(f"Loading model state from '{model_path}'")
            checkpoint = torch.load(model_path)
            # model
            copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            # detail model
            if 'E_detail' in checkpoint.keys():
                copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
                copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
            # training state
            self.start_epoch = 0  # checkpoint['epoch']
            self.start_iter = 0  # checkpoint['iter']
        else:
            print('Start training from scratch')
            self.start_epoch = 0
            self.start_iter = 0

    def visualize(self, visdict, savepath, catdim=1):
        grids = {}
        for key in visdict:
            if visdict[key] is None:
                continue
            grids[key] = torchvision.utils.make_grid(
                func.interpolate(visdict[key], [self.config.image_size, self.config.image_size])).detach().cpu()
        grid = torch.cat(list(grids.values()), catdim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath is not None:
            cv2.imwrite(savepath, grid_image)
        return grid_image


class ExpDECA(DECA):
    """
    This is the EMOCA class (previously ExpDECA). This class derives from DECA and add EMOCA-related functionality.
    Such as a separate expression decoder and related.
    """

    def __init__(self, config):
        super().__init__(config)
        self.n_exp_param = None

    def _create_model(self):
        # 1) Initialize DECA
        super()._create_model()
        # E_flame should be fixed for expression EMOCA
        self.E_flame.requires_grad_(False)
        # 2) add expression decoder
        if self.config.expression_backbone == 'deca_clone':
            # Clones the original
            # DECA coarse decoder (and the entire decoder will be trainable) - This is in final EMOCA.
            self.E_expression = ResnetEncoder(self.n_exp_param)
            # clone parameters of the ResNet
            self.E_expression.encoder.load_state_dict(self.E_flame.encoder.state_dict())
        else:
            raise ValueError(f"Invalid expression backbone: '{self.config.expression_backbone}'")
        if self.config.get('zero_out_last_enc_layer', False):
            self.E_expression.reset_last_layer()

    def get_coarse_trainable_parameters(self):
        print("Add E_expression.parameters() to the optimizer")
        return list(self.E_expression.parameters())

    def reconfigure(self, config):
        super().reconfigure(config)
        self.n_exp_param = self.config.n_exp
        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            self.n_exp_param += self.config.n_pose
        elif self.config.exp_deca_global_pose or self.config.exp_deca_jaw_pose:
            self.n_exp_param += 3

    def encode_flame(self, images):
        # other regressors have to do a separate pass over the image
        deca_code = super().encode_flame(images)
        exp_deca_code = self.E_expression(images)
        return deca_code, exp_deca_code

    def decompose_code(self, code):
        deca_code = code[0]
        expdeca_code = code[1]
        deca_code_list, _ = super().decompose_code(deca_code)
        exp_idx = 2
        pose_idx = 3
        deca_code_list_copy = deca_code_list.copy()
        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            exp_code = expdeca_code[:, :self.config.n_exp]
            pose_code = expdeca_code[:, self.config.n_exp:]
            deca_code_list[exp_idx] = exp_code
            deca_code_list[pose_idx] = pose_code
        elif self.config.exp_deca_global_pose:
            # global pose from ExpDeca, jaw pose from EMOCA
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_exp_deca, pose_code_deca[:, 3:]], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        elif self.config.exp_deca_jaw_pose:
            # global pose from EMOCA, jaw pose from ExpDeca
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_deca[:, :3], pose_code_exp_deca], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        else:
            exp_code = expdeca_code
            deca_code_list[exp_idx] = exp_code
        return deca_code_list, deca_code_list_copy

    def train(self, mode: bool = True):
        super().train(mode)
        # for expression deca, we are not training
        # the resnet feature extractor plus the identity/light/texture regressor
        self.E_flame.eval()
        if mode:
            if self.mode == DecaMode.COARSE:
                self.E_expression.train()
                self.E_detail.eval()
                self.D_detail.eval()
            elif self.mode == DecaMode.DETAIL:
                if self.config.train_coarse:
                    self.E_expression.train()
                else:
                    self.E_expression.eval()
                self.E_detail.train()
                self.D_detail.train()
            else:
                raise ValueError(f"Invalid mode '{self.mode}'")
        else:
            self.E_expression.eval()
            self.E_detail.eval()
            self.D_detail.eval()
        return self


def instantiate_deca(cfg, stage, prefix, checkpoint=None, checkpoint_kwargs=None):
    """
    Function that instantiates a DecaModule from checkpoint or config
    """
    if checkpoint is None:
        deca = DecaModule(cfg.model, cfg.learning, cfg.inout, prefix)
        if cfg.model.resume_training:
            # This load the DECA model weights from the original DECA release
            print("[WARNING] Loading EMOCA checkpoint pretrained by the old code")
            deca.deca.load_old_checkpoint()
    else:
        checkpoint_kwargs = checkpoint_kwargs or {}
        deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
        if stage == 'train':
            mode = True
        else:
            mode = False
        deca.reconfigure(cfg.model, cfg.inout, cfg.learning, prefix, downgrade_ok=True, train=mode)
    return deca
