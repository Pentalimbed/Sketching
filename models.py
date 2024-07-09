import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Flatten, Linear, Sequential, Hardsigmoid, LeakyReLU, \
    AvgPool2d


class SegmentDecoder(torch.nn.Module):
    def __init__(self, in_dims, n_channel=3, prim_num=1):
        super().__init__()

        self.prim_num = prim_num
        self.n_channel = n_channel

        self.block = Sequential(
            Linear(in_dims, prim_num * (5 + n_channel)),  # start, end, thickness, colour
            Hardsigmoid())

    def forward(self, x):
        n = x.shape[0]

        o = self.block(x)
        o = o.view(n, self.prim_num, 5 + self.n_channel)
        return o


class AgentConvBlock(torch.nn.Module):
    def __init__(self, nin, nout, ksize=3):
        super(AgentConvBlock, self).__init__()
        self.conv1 = Conv2d(nin, nout, ksize, padding=1)
        self.lrelu1 = LeakyReLU(0.2)
        self.conv2 = Conv2d(nout, nout, ksize, padding=1)
        self.lrelu2 = LeakyReLU(0.2)
        self.pool = AvgPool2d(2)

    def forward(self, x):
        h = self.lrelu1(self.conv1(x))
        h = self.lrelu2(self.conv2(h))
        return self.pool(h)


class AgentConvBlockBN(torch.nn.Module):
    def __init__(self, nin, nout, ksize=3):
        super(AgentConvBlockBN, self).__init__()
        self.conv1 = Conv2d(nin, nout, ksize, padding=1)
        self.bn1 = BatchNorm2d(nout)
        self.lrelu1 = LeakyReLU(0.2)
        self.conv2 = Conv2d(nout, nout, ksize, padding=1)
        self.bn2 = BatchNorm2d(nout)
        self.lrelu2 = LeakyReLU(0.2)
        self.pool = AvgPool2d(2)

    def forward(self, x):
        h = self.lrelu1(self.bn1(self.conv1(x)))
        h = self.lrelu2(self.bn2(self.conv2(h)))
        return self.pool(h)


class AgentCNN(torch.nn.Module):
    """
    The AgentCNN network from the SketchNet paper. By default takes a [C, 256, 256] image and projects into a 256 dim
    vector.

    Args:
        channels: number of channels in input
    """

    def __init__(self, channels, batchnorm):
        super(AgentCNN, self).__init__()
        if batchnorm:
            acb = AgentConvBlockBN
        else:
            acb = AgentConvBlock

        self.down1 = acb(channels, 16)
        self.down2 = acb(16, 32)
        self.down3 = acb(32, 64)
        self.down4 = acb(64, 128)
        self.down5 = acb(128, 256)

    @staticmethod
    def _output_size(input_size):
        return int((input_size / 32) ** 2 * 256)

    def forward(self, x):
        h = self.down1(x)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.down5(h)

        return h.view(h.shape[0], -1)


class StrokePredictor(torch.nn.Module):
    def __init__(self, hidden_dims=(64, 256), in_channels=6, n_channel=3, prim_num=1, prim_type='seg'):
        super().__init__()

        self.encoder = AgentCNN(in_channels, True)

        self.decoder_common = Sequential(
            Linear(16384, hidden_dims[0]),
            ReLU(),
            Linear(hidden_dims[0], hidden_dims[1]),
            ReLU(),
        )

        match prim_type:
            case 'seg':
                self.decoder_prim = SegmentDecoder(hidden_dims[1], n_channel, prim_num)
            case _:
                raise RuntimeError("Invalid primitive type!")

    def forward(self, x):
        o = self.encoder(x)
        o = self.decoder_common(o)
        o = self.decoder_prim(o)
        return o
