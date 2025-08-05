import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class PSNRLoss(torch.nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class PositiveContrastLoss(nn.Module):
    def __init__(self):
        super(PositiveContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.layer_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.neg_weights = [1.0, 0.25, 0.125]  # [1.0, 1.0, 1.0]  #

    # p1, p2作为正样本
    def forward(self, a, p0, p1, p2, n):
        """
        Parameters
        ----------
        a: anchor->the image restored at stage 3
        p0: positive->paired high definition image  (strong-positive)
        p1: the image restored at stage 2  (normal-positive)
        p2: the image restored at stage 1  (weak-positive)
        n: paired low-quality image

        Returns
        -------
        """
        a_vgg, p0_vgg, p1_vgg, p2_vgg, n_vgg = self.vgg(a), self.vgg(p0), self.vgg(p1), self.vgg(p2), self.vgg(n)
        loss = 0

        # positive: p0a, ap1, p1p2, negative: an
        for i in range(len(a_vgg)):
            d_ap0 = self.l1(a_vgg[i], p0_vgg[i].detach())
            d_ap1 = self.l1(a_vgg[i], p1_vgg[i])
            d_ap2 = self.l1(a_vgg[i], p2_vgg[i])

            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            d_ap = self.neg_weights[0] * d_ap0 + self.neg_weights[1] * d_ap1 + self.neg_weights[2] * d_ap2
            contrastive = d_ap / (d_an + 1e-7)

            loss += self.layer_weights[i] * contrastive

        return loss


class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()
        # self.l1 = CharbonnierLoss().cuda()

    def forward(self, target, restored, vars_):
        s = torch.exp(-vars_)
        sr_ = torch.mul(restored, s)
        hr_ = torch.mul(target, s)
        loss_uncertainty = self.l1(sr_, hr_) + 2 * torch.mean(vars_)

        return loss_uncertainty


class PerceptualLoss2(nn.Module):
    def __init__(self):
        super(PerceptualLoss2, self).__init__()
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        vgg = models.vgg19(pretrained=True).eval()
        self.loss_net1 = nn.Sequential(*list(vgg.features)[:1]).eval()
        self.loss_net3 = nn.Sequential(*list(vgg.features)[:3]).eval()

    def forward(self, x, y):
        loss1 = self.L1(self.loss_net1(x), self.loss_net1(y))
        loss3 = self.L1(self.loss_net3(x), self.loss_net3(y))
        loss = 0.5 * loss1 + 0.5 * loss3
        return loss
