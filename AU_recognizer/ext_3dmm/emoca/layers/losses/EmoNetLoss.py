import copy
from pathlib import Path
import torch
import torch.nn.functional as func

from .BarlowTwins import BarlowTwinsLossHeadless, BarlowTwinsLoss
from .EmonetLoader import get_emonet
from .Metrics import get_metric
from ...utils.other import class_from_str

try:
    from ...models.EmoNetModule import EmoNetModule
except ImportError as e:
    raise ImportError(f"Could not import EmoNetModule. EmoNet models will not be available."
          f" Make sure you pull the repository with submodules to enable EmoNet.")


def create_emo_loss(device, emoloss=None, trainable=False, dual=False, normalize_features=False, emo_feat_loss=None):
    if emoloss is None:
        return EmoNetLoss(device, emonet=emoloss)
    if isinstance(emoloss, str):
        path = Path(emoloss)
        if path.is_dir():
            from .emotion_loss_loader import emo_network_from_path
            emo_loss = emo_network_from_path(path)
            if isinstance(emo_loss, EmoNetModule):
                emonet = emo_loss.emonet
                print("Creating EmoNetLoss")
                return EmoNetLoss(device, emonet=emonet, trainable=trainable,
                                  normalize_features=normalize_features, emo_feat_loss=emo_feat_loss)
            else:
                if not dual:
                    print(f"Creating EmoBackboneLoss, trainable={trainable}")
                    return EmoBackboneLoss(device, emo_loss, trainable=trainable,
                                           normalize_features=normalize_features, emo_feat_loss=emo_feat_loss)
                else:
                    print(f"Creating EmoBackboneDualLoss")
                    return EmoBackboneDualLoss(device, emo_loss, trainable=trainable, clone_is_trainable=True,
                                               normalize_features=normalize_features, emo_feat_loss=emo_feat_loss)
        else:
            raise ValueError("Please specify the directory which contains the config of the trained Emonet.")


class EmoLossBase(torch.nn.Module):
    def __init__(self, trainable=False, normalize_features=False, emo_feat_loss=None, au_loss=None,
                 last_feature_size=None):
        super().__init__()
        self.last_feature_size = last_feature_size
        if emo_feat_loss is not None:
            if isinstance(emo_feat_loss, str) and 'barlow_twins' in emo_feat_loss:
                emo_feat_loss_type = emo_feat_loss
                emo_feat_loss = {"type": emo_feat_loss_type}
            if isinstance(emo_feat_loss, dict) and 'barlow_twins' in emo_feat_loss["type"]:
                emo_feat_loss["feature_size"] = last_feature_size
            emo_feat_loss = get_metric(emo_feat_loss)
        if isinstance(au_loss, str):
            au_loss = class_from_str(au_loss, func)
        self.emo_feat_loss = emo_feat_loss or func.l1_loss
        self.normalize_features = normalize_features
        self.valence_loss = func.l1_loss
        self.arousal_loss = func.l1_loss
        self.expression_loss = func.l1_loss
        self.au_loss = au_loss or func.l1_loss
        self.input_emotion = None
        self.output_emotion = None
        self.trainable = trainable

    @property
    def input_emo(self):
        return self.input_emotion

    @property
    def output_emo(self):
        return self.output_emotion

    def _forward_input(self, images):
        with torch.no_grad():
            result = self(images)
        return result

    def _forward_output(self, images):
        return self(images)

    def compute_loss(self, input_images, output_images, batch_size=None, ring_size=None):
        input_emotion = self._forward_input(input_images)
        output_emotion = self._forward_output(output_images)
        self.input_emotion = input_emotion
        self.output_emotion = output_emotion
        if 'emo_feat' in input_emotion.keys():
            input_emofeat = input_emotion['emo_feat']
            output_emofeat = output_emotion['emo_feat']
            if self.normalize_features:
                input_emofeat = (input_emofeat / input_emofeat.view(input_images.shape[0], -1)
                                 .norm(dim=1).view(-1, *((len(input_emofeat.shape) - 1) * [1])))
                output_emofeat = (output_emofeat / output_emofeat.view(output_images.shape[0], -1)
                                  .norm(dim=1).view(-1, *((len(input_emofeat.shape) - 1) * [1])))
            if isinstance(self.emo_feat_loss, (BarlowTwinsLossHeadless, BarlowTwinsLoss)):
                emo_feat_loss_1 = self.emo_feat_loss(input_emofeat, output_emofeat, batch_size=batch_size,
                                                     ring_size=ring_size).mean()
            else:
                emo_feat_loss_1 = self.emo_feat_loss(input_emofeat, output_emofeat).mean()
        else:
            emo_feat_loss_1 = None
        input_emofeat_2 = input_emotion['emo_feat_2']
        output_emofeat_2 = output_emotion['emo_feat_2']
        if self.normalize_features:
            input_emofeat_2 = (input_emofeat_2 / input_emofeat_2
                               .view(input_images.shape[0], -1)
                               .norm(dim=1).view(-1, *((len(input_emofeat_2.shape) - 1) * [1])))
            output_emofeat_2 = (output_emofeat_2 / output_emofeat_2
                                .view(output_images.shape[0], -1).norm(dim=1)
                                .view(-1, *((len(input_emofeat_2.shape) - 1) * [1])))
        if isinstance(self.emo_feat_loss, (BarlowTwinsLossHeadless, BarlowTwinsLoss)):
            emo_feat_loss_2 = self.emo_feat_loss(input_emofeat_2, output_emofeat_2, batch_size=batch_size,
                                                 ring_size=ring_size).mean()
        else:
            emo_feat_loss_2 = self.emo_feat_loss(input_emofeat_2, output_emofeat_2).mean()
        if 'valence' in input_emotion.keys() and input_emotion['valence'] is not None:
            valence_loss = self.valence_loss(input_emotion['valence'], output_emotion['valence'])
        else:
            valence_loss = None
        if 'arousal' in input_emotion.keys() and input_emotion['arousal'] is not None:
            arousal_loss = self.arousal_loss(input_emotion['arousal'], output_emotion['arousal'])
        else:
            arousal_loss = None
        if 'expression' in input_emotion.keys() and input_emotion['expression'] is not None:
            expression_loss = self.expression_loss(input_emotion['expression'], output_emotion['expression'])
        elif 'expr_classification' in input_emotion.keys() and input_emotion['expr_classification'] is not None:
            expression_loss = self.expression_loss(input_emotion['expr_classification'],
                                                   output_emotion['expr_classification'])
        else:
            expression_loss = None
        if 'AUs' in input_emotion.keys() and input_emotion['AUs'] is not None:
            au_loss = self.au_loss(input_emotion['AUs'], output_emotion['AUs'])
        else:
            au_loss = None
        return emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss, au_loss

    def get_trainable_params(self):
        params = []
        if isinstance(self.emo_feat_loss, (BarlowTwinsLossHeadless, BarlowTwinsLoss)):
            params += list(self.emo_feat_loss.parameters())
        return params

    def is_trainable(self):
        return len(self.get_trainable_params()) != 0

    def train(self, b=True):
        super().train(False)
        if isinstance(self.emo_feat_loss, (BarlowTwinsLossHeadless, BarlowTwinsLoss)):
            self.emo_feat_loss.train(b)
        return self


class EmoNetLoss(EmoLossBase):
    def __init__(self, device, emonet=None, trainable=False, normalize_features=False, emo_feat_loss=None,
                 au_loss=None):
        if emonet is None:
            emonet = get_emonet(device).eval()
        last_feature_size = 256
        if isinstance(emo_feat_loss, dict) and "barlow_twins" in emo_feat_loss["type"]:
            # if barlow twins, we need to know the feature size
            emo_feat_loss["feature_size"] = last_feature_size
        super().__init__(trainable, normalize_features=normalize_features, emo_feat_loss=emo_feat_loss, au_loss=au_loss,
                         last_feature_size=last_feature_size)
        self.emonet = emonet
        if not trainable:
            self.emonet.eval()
            self.emonet.requires_grad_(False)
        else:
            self.emonet.train()
            self.emonet.emo_parameters_requires_grad(True)
        self.size = (256, 256)

    @property
    def network(self):
        return self.emonet

    def to(self, *args, **kwargs):
        self.emonet = self.emonet.to(*args, **kwargs)

    def eval(self):
        self.emonet = self.emonet.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'emonet'):
            self.emonet = self.emonet.eval()  # evaluation mode no matter what, it's just a loss function

    def forward(self, images):
        return self.emonet_out(images)

    def emonet_out(self, images):
        images = func.interpolate(images, self.size, mode='bilinear')
        return self.emonet(images, intermediate_features=True)

    def get_trainable_params(self):
        if self.trainable:
            return self.emonet.emo_parameters
        return []


class EmoBackboneLoss(EmoLossBase):

    def __init__(self, device, backbone, trainable=False, normalize_features=False, emo_feat_loss=None, au_loss=None):
        if isinstance(emo_feat_loss, dict) and "barlow_twins" in emo_feat_loss["type"]:
            # if barlow twins, we need to know the feature size
            emo_feat_loss["feature_size"] = backbone.get_last_feature_size()

        super().__init__(trainable, normalize_features=normalize_features, emo_feat_loss=emo_feat_loss, au_loss=au_loss,
                         last_feature_size=backbone.get_last_feature_size())
        self.backbone = backbone
        if not trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        else:
            self.backbone.requires_grad_(True)
        self.backbone.to(device)

    def get_trainable_params(self):
        params = super().get_trainable_params()
        if self.trainable:
            params += list(self.backbone.parameters())
        return params

    def forward(self, images):
        # noinspection PyProtectedMember
        return self.backbone._forward(images)

    def train(self, b=True):
        super().train(b)
        if not self.trainable:
            self.backbone.eval()
        else:
            self.backbone.train(b)
        return self


class EmoBackboneDualLoss(EmoBackboneLoss):
    def __init__(self, device, backbone, trainable=False, clone_is_trainable=True,
                 normalize_features=False, emo_feat_loss=None, au_loss=None):
        super().__init__(device, backbone, trainable, normalize_features=normalize_features,
                         emo_feat_loss=emo_feat_loss, au_loss=au_loss)
        assert not trainable

        if not clone_is_trainable:
            raise ValueError("The second cloned backbone (used to be fine tuned on renderings) is not trainable. "
                             "Probably not what you want.")
        self.clone_is_trainable = clone_is_trainable
        self.trainable_backbone = copy.deepcopy(backbone)
        if not clone_is_trainable:
            self.trainable_backbone.requires_grad_(False)
            self.trainable_backbone.eval()
        else:
            self.trainable_backbone.requires_grad_(True)
        self.trainable_backbone.to(device)

    def get_trainable_params(self):
        trainable_params = super().get_trainable_params()
        if self.clone_is_trainable:
            trainable_params += list(self.trainable_backbone.parameters())
        return trainable_params

    def _forward_output(self, images):
        # noinspection PyProtectedMember
        return self.trainable_backbone._forward(images)

    def train(self, b=True):
        super().train(b)
        if not self.clone_is_trainable:
            self.trainable_backbone.eval()
        else:
            self.trainable_backbone.train(b)
        return self
