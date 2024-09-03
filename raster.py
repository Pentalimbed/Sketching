from typing import Any, Tuple

import torch


class DistanceRasterStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d2, thickness):
        ctx.save_for_backward(d2, thickness)
        return (d2 < thickness * thickness * 0.25).float()

    @staticmethod
    def backward(ctx, grad_output):
        d2, thickness = ctx.saved_tensors
        d2 = d2.detach().requires_grad_()
        thickness = thickness.detach().requires_grad_()

        # mult
        # thickness = thickness * 0.7

        sigma = 0.54925 * thickness
        denum = sigma * sigma + torch.finfo().eps
        forward_func = torch.exp(-d2 / denum)

        grad_d = -forward_func / denum
        grad_thickness = 2 * 0.54925 * 0.54925 * d2 * thickness * forward_func / (denum * denum)

        return grad_output * grad_d, grad_output * grad_thickness


def get_thickness_px_range(a, b, dim, device):
    maxdim = max(dim[0], dim[1])
    left = torch.tensor(max(2.0, maxdim * a), device=device)
    right = torch.tensor(max(2.0, maxdim * b), device=device)
    return left, right


class SegmentRasteriser(torch.nn.Module):
    def __init__(self, dims, thickness_range=(0.0, 1.0), straight_through=False, pos_mode=0):
        super().__init__()

        self.dims = torch.tensor(dims)
        self.thickness_range = thickness_range
        # 0 - start / end
        # 1 - start / center
        self.pos_mode = pos_mode

        self.pos = torch.stack(torch.meshgrid(torch.arange(dims[0]), torch.arange(dims[1]), indexing='ij'))
        self.pos = self.pos.view(1, *self.pos.shape)

        self.straight_through = straight_through

    def to(self, device):
        # it doesn't auto convert
        retval = super().to(device)
        retval.dims = self.dims.to(device)
        retval.pos = self.pos.to(device)

        return retval

    def forward(self, x: torch.Tensor):
        if self.pos_mode == 0:
            start = x[:, :, :2] * self.dims.view(1, 1, 2)
            end = x[:, :, 2:4] * self.dims.view(1, 1, 2)
        elif self.pos_mode == 1:
            start = x[:, :, :2] * self.dims.view(1, 1, 2)
            end = (x[:, :, 2:4] * 2 - x[:, :, :2]) * self.dims.view(1, 1, 2)
        thickness = torch.lerp(*get_thickness_px_range(*self.thickness_range, self.dims, x.device), x[:, :, [4]])
        colour = x[:, :, 5:]

        start = start.view(*start.shape, 1, 1)
        end = end.view(*end.shape, 1, 1)
        m = end - start

        ps = self.pos - start
        pe = self.pos - end

        t = torch.sum(ps * m, dim=2, keepdim=True) / (torch.sum(m * m, dim=2, keepdim=True) + torch.finfo().eps)
        patm = self.pos - (start + t * m)
        d2 = ((t <= 0) * torch.sum(ps * ps, dim=2, keepdim=True) +
              (t > 0) * (t < 1) * torch.sum(patm * patm, dim=2, keepdim=True) +
              (t >= 1) * torch.sum(pe * pe, dim=2, keepdim=True))

        thickness = thickness.view(*thickness.shape, 1, 1)
        if self.straight_through:
            raster = DistanceRasterStraightThrough.apply
            canvas = raster(d2, thickness)
        else:
            sigma = 0.54925 * thickness
            canvas = torch.exp(-d2 / (sigma * sigma + torch.finfo().eps))

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
