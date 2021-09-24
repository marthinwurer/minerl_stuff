"""
From https://github.com/AlbertZhangHIT/Hartley-spectral-pooling
"""
"""
MIT License

Copyright (c) 2018 ZhangHao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair


def _spectral_crop(input, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            bottom_left = input[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h - 1):, -(cutoff_freq_w - 1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    top_combined = torch.cat((top_left, top_right), dim=-1)
    bottom_combined = torch.cat((bottom_left, bottom_right), dim=-1)
    all_together = torch.cat((top_combined, bottom_combined), dim=-2)

    return all_together


def _spectral_pad(input2, input, oheight, owidth):
    s = input.shape
    cutoff_freq_h = math.ceil(min(oheight, s[-2]) / 2)
    cutoff_freq_w = math.ceil(min(owidth, s[-1]) / 2)

    pad = torch.zeros(s[0], s[1], oheight, owidth, device=input.device)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):] = input[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            pad[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w] = input[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h - 1):, -(cutoff_freq_w - 1):] = input[:, :, -(cutoff_freq_h - 1):,
                                                                      -(cutoff_freq_w - 1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w] = input[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:] = input[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):] = input[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):] = input[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -cutoff_freq_w:] = input[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    return pad


def DiscreteHartleyTransform(input):
    fft = torch.fft.fft2(input, norm='ortho')
    dht = fft.real - fft.imag
    return dht


class SpectralPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, oheight, owidth, scale):
        ctx.oh = oheight
        ctx.ow = owidth
        ctx.save_for_backward(input, scale)

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(input)

        # frequency cropping
        if scale <= 1:
            all_together = _spectral_crop(dht, oheight, owidth)
        else:
            all_together = _spectral_pad(input, dht, oheight, owidth)

        # inverse Hartley transform
        dht = DiscreteHartleyTransform(all_together)
        return dht

    @staticmethod
    def backward(ctx, grad_output):
        input, scale, = ctx.saved_variables

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(grad_output)
        # frequency padding
        grad_input = _spectral_pad(input, dht, ctx.oh, ctx.ow)
        # inverse Hartley transform
        grad_input = DiscreteHartleyTransform(grad_input)
        return grad_input, None, None


class SpectralPool2d(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale = scale_factor

    def forward(self, input):
        H, W = input.size(-2), input.size(-1)
        oheight, owidth = math.ceil(H * self.scale), math.ceil(W * self.scale)

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(input)

        # frequency cropping
        if self.scale <= 1:
            all_together = _spectral_crop(dht, oheight, owidth)
        else:
            all_together = _spectral_pad(input, dht, oheight, owidth)

        # inverse Hartley transform
        output = DiscreteHartleyTransform(all_together)
        return output



        return SpectralPoolingFunction.apply(input, h, w, self.scale_factor)
