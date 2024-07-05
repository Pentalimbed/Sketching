import torch


class SegmentRasteriser(torch.nn.Module):
    def __init__(self, dims, inference=False):
        super().__init__()

        self.pos = torch.stack(torch.meshgrid(torch.arange(dims[0]), torch.arange(dims[1]), indexing='ij'))
        self.pos = self.pos.view(1, *self.pos.shape)

        self.inference = inference

    def forward(self, x):
        start = x[:, :, :2]
        end = x[:, :, 2:4]
        thickness = x[:, :, [4]]
        colour = x[:, :, 5:]

        start = start.view(*start.shape, 1, 1)
        end = end.view(*end.shape, 1, 1)
        m = end - start

        t = torch.norm((self.pos - start) * m, dim=2, keepdim=True) / torch.norm(m * m, dim=2, keepdim=True)
        d = ((t <= 0) * torch.norm(self.pos - start, dim=2, keepdim=True) +
             (t > 0) * (t < 1) * torch.norm(self.pos - (start + t * m), dim=2, keepdim=True) +
             (t >= 1) * torch.norm(self.pos - end, dim=2, keepdim=True))

        thickness = thickness.view(*thickness.shape, 1, 1)
        if self.inference:
            canvas = d * d < thickness * thickness * 0.25
        else:
            sigma = 0.54925 * thickness
            canvas = torch.exp(-d * d / (sigma * sigma))

        return canvas * colour.view(*colour.shape, 1, 1)


def _composite_over(imgs: torch.Tensor) -> torch.Tensor:
    """
    :param imgs: [stack_size, C, H, W], stack of images to overlay with, higher indices are drawn first
    """
    linv = (1 - imgs) + 1e-10  # .clamp(0)
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
    linv = (1 - imgs[:, [-1], :, :]) + 1e-10  # .clamp(0)
    linvrasters = linv.log()
    vis = (linvrasters.cumsum(dim=0) - linvrasters).exp()
    o = (imgs[:, :-1] * vis).sum(dim=0)
    return o


composite_over_alpha = torch.vmap(_composite_over_alpha)

if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    dims = torch.tensor([64, 54])

    canvas = SegmentRasteriser(dims)(
        torch.cat([
            torch.tensor([[[0.25, 0.25], [0.75, 0.25]]]) * dims.view([1, 1, 2]),
            torch.tensor([[[0.75, 0.75], [0.25, 0.75]]]) * dims.view([1, 1, 2]),
            torch.tensor([[[2], [5]]]),
            torch.tensor([[[1, 0, 0, 1], [0, 0, 1, 1]]])],
            dim=-1)
    )
    canvas = torch.cat([canvas, torch.ones([canvas.shape[0], 1, *canvas.shape[2:]]) * 0.5], dim=1)
    canvas = composite_over_alpha(canvas)

    plt.imshow(torchvision.transforms.ToPILImage()(canvas[0]))
    plt.show()
    plt.waitforbuttonpress()
