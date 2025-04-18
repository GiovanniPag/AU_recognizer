import torch.nn as nn
import torch
import torch.nn.functional as func
from functools import reduce
import torchvision.models as models

from .BarlowTwins import BarlowTwinsLossHeadless, BarlowTwinsLoss
from .FRNet import resnet50, load_state_dict


class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()  # .cuda()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x / self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        return out


class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.patches_OIHW = None
        self.cs_NCHW = None
        self.style_loss = None
        self.content_loss = None
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    @staticmethod
    def sum_normalize(featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    @staticmethod
    def compute_relative_distances(cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        mean_t = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - mean_t, tar - mean_t

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        batch_size = tar.size(0)

        for i in range(batch_size):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_oihw = self.patch_extraction(tar_feat_i)

            cosine_dist_i = func.conv2d(gen_feat_i, patches_oihw)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        #  gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for
                           layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
                             for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss

    def train(self, b=True):
        # there is nothing trainable about this loss
        return super().train(False)


class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_checkpoint_path=None, metric='cosine_similarity', trainable=False):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval()
        checkpoint = pretrained_checkpoint_path or '/ps/scratch/rdanecek/FaceRecognition/resnet50_ft_weight.pkl'
        load_state_dict(self.reg_model, checkpoint)
        # this mean needs to be subtracted from the input images if using the model above
        self.register_buffer('mean_bgr', torch.tensor([91.4953, 103.8827, 131.0912]))
        self.trainable = trainable
        if metric is None:
            metric = 'cosine_similarity'
        if metric not in ["l1", "l1_loss", "l2", "mse", "mse_loss", "cosine_similarity",
                          "barlow_twins", "barlow_twins_headless"]:
            raise ValueError(f"Invalid metric for face recognition feature loss: {metric}")
        if metric == "barlow_twins_headless":
            feature_size = self.reg_model.fc.in_features
            self.bt_loss = BarlowTwinsLossHeadless(feature_size)
        elif metric == "barlow_twins":
            feature_size = self.reg_model.fc.in_features
            self.bt_loss = BarlowTwinsLoss(feature_size)
        else:
            self.bt_loss = None
        self.metric = metric

    def get_trainable_params(self):
        params = []
        if self.trainable:
            params += list(self.reg_model.parameters())
        if self.bt_loss is not None:
            params += list(self.bt_loss.parameters())
        return params

    def train(self, b=True):
        if not self.trainable:
            ret = super().train(False)
        else:
            ret = super().train(b)
        if self.bt_loss is not None:
            self.bt_loss.train(b)
        return ret

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(False)  # face recognition net always frozen
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(requires_grad)

    def freeze_nontrainable_layers(self):
        if not self.trainable:
            super().requires_grad_(False)
        else:
            super().requires_grad_(True)
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(True)

    def reg_features(self, x):
        margin = 10
        x = x[:, :, margin:224 - margin, margin:224 - margin]
        x = func.interpolate(x * 2. - 1., [224, 224], mode='bilinear')
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # input images in RGB in range [0-1] but the network expects them in BGR  [0-255] with subtracted mean_bgr
        img = img[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) * 255 - self.mean_bgr
        img = img.permute(0, 3, 1, 2)
        return img

    @staticmethod
    def _cos_metric(x1, x2):
        return 1.0 - func.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, batch_size=None, ring_size=None):
        gen = self.transform(gen)
        tar = self.transform(tar)
        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        if self.metric == "cosine_similarity":
            loss = self._cos_metric(gen_out, tar_out).mean()
        elif self.metric in ["l1", "l1_loss", "mae"]:
            loss = torch.nn.functional.l1_loss(gen_out, tar_out)
        elif self.metric in ["mse", "mse_loss", "l2", "l2_loss"]:
            loss = torch.nn.functional.mse_loss(gen_out, tar_out)
        elif self.metric in ["barlow_twins_headless", "barlow_twins"]:
            loss = self.bt_loss(gen_out, tar_out, batch_size=batch_size, ring_size=ring_size)
        else:
            raise ValueError(f"Invalid metric for face recognition feature loss: {self.metric}")
        return loss
