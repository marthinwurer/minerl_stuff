"""
Taken from https://github.com/walsvid/CoordConv

License:

MIT License

Copyright (c) 2018 Walsvid

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
import functools

import torch
import torch.nn as nn
import torch.nn.modules.conv as conv


# @functools.cache
def build_coords(rank, shape, with_r=False):
    if rank == 1:
        batch_size_shape, channel_in_shape, dim_x = shape
        xx_range = torch.arange(dim_x, dtype=torch.int32)
        xx_channel = xx_range[None, None, :]

        xx_channel = xx_channel.float() / (dim_x - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

        out = [xx_channel]

        if with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
            out.append(rr)

    elif rank == 2:
        batch_size_shape, channel_in_shape, dim_y, dim_x = shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        out = [xx_channel, yy_channel]

        if with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out.append(rr)

    elif rank == 3:
        batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = shape
        xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
        zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

        xy_range = torch.arange(dim_y, dtype=torch.int32)
        xy_range = xy_range[None, None, None, :, None]

        yz_range = torch.arange(dim_z, dtype=torch.int32)
        yz_range = yz_range[None, None, None, :, None]

        zx_range = torch.arange(dim_x, dtype=torch.int32)
        zx_range = zx_range[None, None, None, :, None]

        xy_channel = torch.matmul(xy_range, xx_ones)
        xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

        yz_channel = torch.matmul(yz_range, yy_ones)
        yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
        yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

        zx_channel = torch.matmul(zx_range, zz_ones)
        zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
        zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

        out = [xx_channel, yy_channel, zz_channel]

        if with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                            torch.pow(yy_channel - 0.5, 2) +
                            torch.pow(zz_channel - 0.5, 2))
            out.append(rr)

    else:
        raise NotImplementedError

    return out


class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        coords = [coord.to(input_tensor.device) for coord in build_coords(self.rank, input_tensor.shape)]
        out = torch.cat([input_tensor, *coords], dim=1)
        return out


class CoordConv1d(conv.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv3d(conv.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 3
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv3d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConvTranspose2d(conv.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size,
                                                   stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.ConvTranspose2d(in_channels + self.rank + int(with_r), out_channels,
                                       kernel_size, stride, padding, output_padding=0,
                                       dilation=dilation, groups=groups, bias=bias)

    def forward(self, input, output_size=None):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input)
        out = self.conv(out)

        return out
