import torch
from torch import nn


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        self.act = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)

    def forward(self, x):
        res = self.act(self.conv1(x))
        res += x
        res = self.conv2(res)
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class PatchAttention(nn.Module):
    def __init__(self, channel, reduction, pa_size):
        super(PatchAttention, self).__init__()
        self.pa_size = pa_size
        self.avg_pool = nn.AdaptiveAvgPool2d(pa_size)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y1 = self.conv_du(self.avg_pool(x)).unsqueeze(3).unsqueeze(-1)
        # b, c, self.pa_size[0], h // self.pa_size[0], self.pa_size[1], h // self.pa_size[1]
        x_partition = x.view(b, c, self.pa_size[0], h // self.pa_size[0], self.pa_size[1], w // self.pa_size[1])
        att = x_partition * y1
        return att.view(b, c, h, w)  # (B, C, H, W)


class PAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, pa_size):
        super(PAB, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        self.act = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)

        self.PA = PatchAttention(n_feat, reduction, pa_size=pa_size)

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        res = self.PA(res)
        res += x
        return res


# Pixel-wise Refine Unit
class PRU(nn.Module):
    def __init__(self, n_feat, reduction):
        super(PRU, self).__init__()
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False)

        self.fr = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // reduction, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(n_feat // reduction, n_feat, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.conv(x)
        att = self.fr(res)

        return att * res + x


class PAG(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, num_cab, pa_sizes):
        super(PAG, self).__init__()
        self.body = nn.Sequential(
            *[PAB(n_feat, kernel_size, reduction, bias=bias, pa_size=pa_sizes[i]) for i in range(num_cab)],
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        )

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, 3, kernel_size, padding=kernel_size // 2, bias=bias)
        self.conv3 = nn.Conv2d(3, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


# Uncertainty Supervised Attention Unit
class USAU(nn.Module):
    def __init__(self, in_channels, n_feats, out_channels, kernel, bias):
        super(USAU, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, n_feats, kernel_size=kernel, padding=kernel // 2, bias=bias)
        self.conv_out = nn.Conv2d(n_feats, out_channels, kernel_size=kernel, padding=kernel // 2, bias=bias)
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=bias)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=bias)

        self.var_est = nn.Sequential(*[nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True)])
        self.var_map = nn.Sequential(*[nn.Conv2d(n_feats, out_channels, kernel_size=3, padding=1), nn.LeakyReLU(0.1, inplace=True)])
        self.att_map = nn.Sequential(*[nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0), nn.Sigmoid()])

    def forward(self, x, img, last=False):
        fusion = self.conv1(x) + self.conv_in(img)

        restored = self.conv_out(fusion) + img
        var_est = self.var_est(fusion)
        var_map = self.var_map(var_est)
        att_map = self.att_map(var_est)

        out_feat = (att_map * self.conv2(x) + x) if not last else None

        return out_feat, restored, var_map


class SubNet1(nn.Module):
    def __init__(self, num_blocks, pa, n_feat, kernel_size, reduction, feats_step, bias):
        super(SubNet1, self).__init__()

        self.encoder1 = nn.Sequential(*[PAB(n_feat, kernel_size, reduction, bias, pa[0]) for _ in range(num_blocks[0])])
        self.encoder2 = nn.Sequential(*[PAB(n_feat + feats_step, kernel_size, reduction, bias, pa[1]) for _ in range(num_blocks[1])])
        self.encoder3 = nn.Sequential(*[PAB(n_feat + (feats_step * 2), kernel_size, reduction, bias, pa[2]) for _ in range(num_blocks[2])])

        self.decoder3 = nn.Sequential(*[PAB(n_feat + (feats_step * 2), kernel_size, reduction, bias, pa[2]) for _ in range(num_blocks[2])])
        self.decoder2 = nn.Sequential(*[PAB(n_feat + feats_step, kernel_size, reduction, bias, pa[1]) for _ in range(num_blocks[1])])
        self.decoder1 = nn.Sequential(*[PAB(n_feat, kernel_size, reduction, bias, pa[0]) for _ in range(num_blocks[0])])

        self.down1 = DownSample(n_feat, feats_step)
        self.down2 = DownSample(n_feat + feats_step, feats_step)

        self.up1 = UpSample(n_feat, feats_step)
        self.up2 = UpSample(n_feat + feats_step, feats_step)

        self.skipatt2 = PAB(n_feat + feats_step, kernel_size, reduction, bias, pa_size=pa[1])
        self.skipatt1 = PAB(n_feat, kernel_size, reduction, bias, pa_size=pa[0])

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.down1(e1))
        e3 = self.encoder3(self.down2(e2))

        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.up2(d3) + self.skipatt2(e2))
        d1 = self.decoder1(self.up1(d2) + self.skipatt1(e1))

        return d1


class SubNet2(nn.Module):
    def __init__(self, num_blocks, pa, n_feat, kernel_size, reduction, feats_step, bias):
        super(SubNet2, self).__init__()

        self.encoder1 = nn.Sequential(*[PAB(n_feat, kernel_size, reduction, bias, pa[0]) for _ in range(num_blocks[0])])
        self.encoder2 = nn.Sequential(*[PAB(n_feat + feats_step, kernel_size, reduction, bias, pa[1]) for _ in range(num_blocks[1])])

        self.decoder2 = nn.Sequential(*[PAB(n_feat + feats_step, kernel_size, reduction, bias, pa[1]) for _ in range(num_blocks[1])])
        self.decoder1 = nn.Sequential(*[PAB(n_feat, kernel_size, reduction, bias, pa[0]) for _ in range(num_blocks[0])])

        self.down1 = DownSample(n_feat, feats_step)

        self.up1 = UpSample(n_feat, feats_step)

        self.skipatt1 = PAB(n_feat, kernel_size, reduction, bias, pa_size=pa[0])

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.down1(e1))

        d2 = self.decoder2(e2)
        d1 = self.decoder1(self.up1(d2) + self.skipatt1(e1))

        return d1


class SubNet3(nn.Module):
    def __init__(self, num_pab, pa, n_feat, kernel_size, reduction, bias):
        super(SubNet3, self).__init__()
        self.PAG1 = PAG(n_feat, kernel_size, reduction, bias, num_pab, pa)
        self.PAG2 = PAG(n_feat, kernel_size, reduction, bias, num_pab, pa)
        self.PAG3 = PAG(n_feat, kernel_size, reduction, bias, num_pab, pa)
        self.PAG4 = PAG(n_feat, kernel_size, reduction, bias, num_pab, pa)

    def forward(self, x):
        x = self.PAG1(x)
        x = self.PAG2(x)
        x = self.PAG3(x)
        x = self.PAG4(x)

        return x


class DSRIR(nn.Module):
    def __init__(self, in_c=3, out_c=3, sn1_feats=32, sn2_feats=48, sn3_feats=64, feats_step1=32, feats_step2=16, kernel_size=3, reduction=8, bias=False):
        super(DSRIR, self).__init__()
        sn1_pab = [2, 2, 2]
        sn2_pab = [2, 4]
        sn3_pab = 4
        sn1_pa = [(8, 8), (4, 4), (2, 2)]
        sn2_pa = [(4, 4), (2, 2)]
        sn3_pa = [(8, 8), (4, 4), (2, 2), (1, 1)]
        self.sn1_feats = sn1_feats
        self.sn2_feats = sn2_feats
        self.sn3_feats = sn3_feats

        self.ebd = nn.Conv2d(in_c, sn1_feats + sn2_feats + sn3_feats, kernel_size, padding=kernel_size // 2, bias=bias)

        self.SN1 = SubNet1(sn1_pab, sn1_pa, sn1_feats, kernel_size, reduction, feats_step1, bias)
        self.PRU1 = PRU(sn1_feats, reduction)
        self.usau_1 = USAU(in_c, sn1_feats, out_c, kernel=3, bias=bias)

        self.channel_comp1 = nn.Conv2d(sn1_feats + sn2_feats, sn2_feats, kernel_size=1, padding=0, bias=bias)
        self.SN2 = SubNet2(sn2_pab, sn2_pa, sn2_feats, kernel_size, reduction, feats_step2, bias)
        self.PRU2 = PRU(sn2_feats, reduction)
        self.usau_2 = USAU(in_c, sn2_feats, out_c, kernel=3, bias=bias)

        self.channel_comp2 = nn.Conv2d(sn2_feats + sn3_feats, sn3_feats, kernel_size=1, padding=0, bias=bias)
        self.SN3 = SubNet3(sn3_pab, sn3_pa, sn3_feats, kernel_size, reduction, bias)
        self.PRU3 = PRU(sn3_feats, reduction)
        self.usau_3 = USAU(in_c, sn3_feats, out_c, kernel=3, bias=bias)

    def forward(self, img):
        x = self.ebd(img)
        x1, x2, x3 = torch.split(x, [self.sn1_feats, self.sn2_feats, self.sn3_feats], dim=1)

        x = self.SN1(x1)
        x = self.PRU1(x)
        x, img1, var1 = self.usau_1(x, img)

        x = self.channel_comp1(torch.cat([x, x2], dim=1))
        x = self.SN2(x)
        x = self.PRU2(x)
        x, img2, var2 = self.usau_2(x, img)

        x = self.channel_comp2(torch.cat([x, x3], dim=1))
        x = self.SN3(x)
        x = self.PRU3(x)
        _, img3, var3 = self.usau_3(x, img, last=True)

        return [img3, img2, img1], [var3, var2, var1]


if __name__ == '__main__':
    from copy import deepcopy
    from thop import profile

    def model_info(model, verbose=False, img_size=128):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in model.parameters())  # number parameters
        # n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
        if verbose:
            print('%5s %40s %9s %12s %20s %10s %10s' % (
                'layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
            for i, (name, p) in enumerate(model.named_parameters()):
                name = name.replace('module_list.', '')
                print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                      (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

        stride = 32
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1e9  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = '%.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS

        # {n_g / 1e6:.2f}MB gradients
        print(f"Model Summary: Layers: {len(list(model.modules()))}, Parameters: {n_p / 1e6:.2f}MB,  FLOPs: {fs}")


    model = DSRIR().cuda()
    model_info(model, False, img_size=256)
