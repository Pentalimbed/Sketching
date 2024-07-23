import torch


class SegmentRasteriser(torch.nn.Module):
    def __init__(self, dims, inference=False):
        super().__init__()

        self.dims = torch.tensor(dims)

        self.pos = torch.stack(torch.meshgrid(torch.arange(dims[0]), torch.arange(dims[1]), indexing='ij'))
        self.pos = self.pos.view(1, *self.pos.shape)

        self.inference = inference

    def to(self, device):
        # it doesn't auto convert
        self.dims = self.dims.to(device)
        self.pos = self.pos.to(device)

        return super().to(device)

    def forward(self, x: torch.Tensor):
        start = x[:, :, :2] * self.dims.view(1, 1, 2)
        end = x[:, :, 2:4] * self.dims.view(1, 1, 2)
        thickness = torch.lerp(torch.tensor(1.0, device=x.device), torch.max(self.dims) * 0.5, x[:, :, [4]])
        colour = x[:, :, 5:]

        start = start.view(*start.shape, 1, 1)
        end = end.view(*end.shape, 1, 1)
        m = end - start

        ps = self.pos - start
        pe = self.pos - end

        t = torch.sum(ps * m, dim=2, keepdim=True) / (torch.sum(m * m, dim=2, keepdim=True) + torch.finfo().eps)
        patm = self.pos - (start + t * m)
        d = ((t <= 0) * torch.sum(ps * ps, dim=2, keepdim=True) +
             (t > 0) * (t < 1) * torch.sum(patm * patm, dim=2, keepdim=True) +
             (t >= 1) * torch.sum(pe * pe, dim=2, keepdim=True))

        thickness = thickness.view(*thickness.shape, 1, 1)
        if self.inference:
            canvas = d * d < thickness * thickness * 0.25
        else:
            sigma = 0.54925 * thickness
            canvas = torch.exp(-d * d / (sigma * sigma + torch.finfo().eps))

        return canvas * colour.view(*colour.shape, 1, 1)


def _composite_softor(imgs: torch.Tensor) -> torch.Tensor:
    linv = 1 - imgs
    return 1 - torch.prod(linv, dim=0)


composite_softor = torch.vmap(_composite_softor)


def _composite_over(imgs: torch.Tensor) -> torch.Tensor:
    """
    :param imgs: [stack_size, C, H, W], stack of images to overlay with, higher indices are drawn first
    """
    linv = (1 - imgs) + torch.finfo().eps
    linvrasters = linv.log()
    vis = (linvrasters.cumsum(dim=0) - linvrasters).exp()
    o = (imgs * vis).sum(dim=0)
    return o


composite_over = torch.vmap(_composite_over)


def _composite_over_alpha(imgs: torch.Tensor) -> torch.Tensor:
    """
    :param imgs: [stack_size, C, H, W], stack of images to overlay with, higher indices are drawn first.
        Alpha in the last channel, the first image (canvas) is fixed to an alpha of 1
    """
    linv = (1 - imgs[:, [-1], :, :]) + torch.finfo().eps
    linvrasters = linv.log()
    linvrasters_sum = linvrasters.cumsum(dim=0)

    vis = (imgs[:, [-1], :, :] + torch.finfo().eps).log()
    vis[1:] += linvrasters_sum[:-1]
    vis = vis.exp()
    o = (imgs[:, :-1] * vis).sum(dim=0)
    return o


composite_over_alpha = torch.vmap(_composite_over_alpha)

if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    dims = torch.tensor([64, 54])

    canvas = SegmentRasteriser(dims)(
        torch.cat([
            torch.tensor([[[0.25, 0.25], [0.75, 0.25]]]),
            torch.tensor([[[0.75, 0.75], [0.25, 0.75]]]),
            torch.tensor([[[2], [5]]]) / torch.max(dims.view(1, 1, 2)),
            torch.tensor([[[1], [1]]])],
            dim=-1)
    )
    canvas = torch.cat([torch.tensor([[[1, 0, 0], [0, 0, 1]]]).view(1, 2, 3, 1, 1).expand(1, 2, 3, *canvas.shape[-2:]),
                        canvas], dim=2)
    canvas = torch.cat([canvas, torch.ones([canvas.shape[0], 1, *canvas.shape[2:]]) * 0.5], dim=1)
    canvas = _composite_over_alpha(canvas[0])

    plt.imshow(torchvision.transforms.ToPILImage()(canvas))
    plt.waitforbuttonpress()
    plt.close()
