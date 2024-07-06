import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Flatten, Linear, Sequential, Hardsigmoid


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


class StrokePredictor(torch.nn.Module):
    def __init__(self, latent_dim=1024, hidden_dims=(64, 256), in_channels=6, n_channel=3, prim_num=1, prim_type='seg'):
        super().__init__()

        self.encoder = Sequential(
            Conv2d(in_channels=in_channels, out_channels=64,
                   kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64,
                   kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64,
                   kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64,
                   kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=64),
            ReLU(),
            AdaptiveAvgPool2d(output_size=8),
            Flatten(),
            Linear(4096, latent_dim)
        )

        self.decoder_common = Sequential(
            Linear(latent_dim, hidden_dims[0]),
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
