"""pix2pix/CycleGAN-style U-Net (WITH skip connections), vendored config-free from
baseline/models/unet.py, for a fair 64x128 comparison group vs the no-skip A9 decoder.

Returns the same dict interface as FullMapNet: {"D","D0","extras"}.
"""
import functools
import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True); downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True); upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Sigmoid()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=use_bias)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=use_bias)
            model = [downrelu, downconv, downnorm] + [submodule] + [uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], 1)


class UNet(nn.Module):
    """input (B,in_ch,H,W) -> sigmoid depth (B,1,H,W). num_downs halves each dim."""

    def __init__(self, cfg):
        super().__init__()
        in_ch = getattr(cfg, "in_ch", 2)
        ngf = getattr(cfg, "ngf", 64)
        num_downs = getattr(cfg, "unet_downs", 6)            # 64x128 -> 1x2 at d=6
        norm = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        blk = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                      norm_layer=norm, innermost=True)
        for _ in range(num_downs - 5):
            blk = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=blk, norm_layer=norm)
        blk = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=blk, norm_layer=norm)
        blk = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=blk, norm_layer=norm)
        blk = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=blk, norm_layer=norm)
        self.model = UnetSkipConnectionBlock(1, ngf, input_nc=in_ch, submodule=blk,
                                             outermost=True, norm_layer=norm)

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        D = self.model(spec)
        return {"D": D, "D0": D, "extras": {}}
