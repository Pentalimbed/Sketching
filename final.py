import argparse
import os
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import lpips

from data import ReconDataset, ImageFolderCustom
from raster import SegmentRasteriser, composite_over_alpha, get_thickness_px_range
from loss import LPLoss


def remap(value, from_range, to_range):
    return (torch.lerp(from_range[0], to_range[1], value) - to_range[0]) / (to_range[1] - to_range[0])


def raw_optimise(
        canvases: torch.Tensor, target: torch.Tensor,
        score_fn: Callable = torch.nn.L1Loss(), loss_fn: Callable = torch.nn.MSELoss(),
        steps=100, thickness_range=(0.0, 1.0), init_length=0.1, init_thickness=1.0, maxdiff_init=False):
    batch_size = canvases.shape[0]
    n_channels = canvases.shape[1]
    img_dims = canvases.shape[2:]
    device = canvases.device

    compositor = composite_over_alpha
    rasteriser = SegmentRasteriser(img_dims, thickness_range)
    rasteriser.to(device)

    one_alphas = torch.ones([batch_size, 1, *img_dims]).to(device)

    with torch.no_grad():
        prims = torch.rand([args.batch_size, args.prims, 5], device=device)
        prims[:, :, [3]] = torch.clamp(prims[:, :, [3]], 1 / 256.0)
        displacements = torch.cat([torch.sin(prims[:, :, [2]] * torch.pi * 2),
                                   torch.cos(prims[:, :, [2]] * torch.pi * 2)],
                                  dim=2) * prims[:, :, [3]] * init_length * 0.5
        if maxdiff_init:
            diff = torch.sum(target - canvases, dim=1)
            argmax_diff = torch.stack([(diff[i] == torch.max(diff[i])).nonzero()[[0]] for i in range(diff.size(0))],
                                      dim=0)
            argmax_diff = argmax_diff / torch.tensor([[img_dims]], device=device)
            prims[:, :, :4] = torch.cat([argmax_diff + displacements, argmax_diff - displacements], dim=2)
        else:
            prims[:, :, :4] = torch.cat([prims[:, :, :2] + displacements, prims[:, :, :2] - displacements], dim=2)
        prims[:, :, 4] *= init_thickness
    prims.requires_grad_()

    colours = torch.rand([args.batch_size, args.prims, n_channels], device=device)
    colours.requires_grad_()

    optim = torch.optim.Adam([prims, colours], lr=0.02)

    for j in range(steps):
        prims.data.clamp_(0, 1)
        prims.grad = None

        colours.data.clamp_(0, 1)
        colours.grad = None

        optim.zero_grad()

        layers = rasteriser(torch.cat([prims, torch.ones([*prims.shape[:2], 1], device=device)], dim=2))

        if j < 1:
            colours.data = torch.sum(layers * target_imgs.unsqueeze(1), dim=(3, 4)) / layers.sum(dim=(3, 4))

        colour_layers = torch.cat([colours.view(*colours.shape[:2], n_channels, 1, 1)
                                  .expand(*colours.shape[:2], n_channels, *layers.shape[-2:]), layers], dim=2)
        colour_layers = torch.cat([colour_layers, torch.cat([canvases, one_alphas], dim=1)
                                  .view([batch_size, 1, n_channels + 1, *img_dims])], dim=1)

        new_canvases = compositor(colour_layers)

        loss = loss_fn(new_canvases, target)
        loss.backward()
        optim.step()

    with torch.no_grad():
        scores = torch.tensor([score_fn(target[i], new_canvases[i]) for i in range(batch_size)], device=device)
    return prims.detach(), colours.detach(), scores, new_canvases.detach()


def inference(target_imgs: torch.Tensor, args,
              score_fn, optimise_fn, error_fn, lpips_fn):
    error_curve = torch.zeros([args.updates, 4])

    canvases = torch.ones([args.batch_size, n_channels, *args.img_dims], device=target_imgs.device)

    initial_error = torch.stack([error_fn(canvases[i], target_imgs[i]) for i in range(args.batch_size)])

    pbar = trange(args.updates, position=tqdm._get_free_pos())
    for u in pbar:
        canvases = canvases.detach()

        prims, colour, scores, new_canvases = raw_optimise(canvases, target_imgs,
                                                           loss_fn=optimise_fn, score_fn=score_fn,
                                                           thickness_range=(0, 1.0), init_thickness=0.05,
                                                           init_length=0.05)

        canvases = new_canvases

        with torch.no_grad():
            error = error_fn(canvases, target_imgs)
            rel_error = (torch.stack(
                [error_fn(canvases[i], target_imgs[i]) for i in range(args.batch_size)]) / initial_error).mean()
            psnr = (10 * torch.log10(
                1 / ((canvases - target_imgs) ** 2 / args.img_dims[0] * args.img_dims[1]).mean(dim=(1, 2, 3))
            )).mean()
            perceptual = lpips_fn(canvases, target_imgs).mean()

            pbar.set_postfix({
                "error": error.cpu().detach().item(),
                "rel_error": rel_error.cpu().detach().item(),
                "psnr": psnr.cpu().detach().item(),
                "perceptual": perceptual.cpu().detach().item()
            })

            error_curve[u, 0] = error
            error_curve[u, 1] = rel_error
            error_curve[u, 2] = psnr
            error_curve[u, 3] = perceptual

            if (args.save_per_stroke > 0 and u % args.save_per_stroke == 0) or u == args.updates - 1:
                img = torchvision.transforms.ToPILImage()(canvases[0])
                img.save(os.path.join(args.output, f"{u}.png"), format=None)

    return canvases, error_curve


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TrainSketch')
    parser.add_argument('-c', '--checkpoint')

    parser.add_argument('-o', '--output', type=str, default="outputs")
    parser.add_argument('--save_per_stroke', type=int, default=0)

    parser.add_argument('--inference', action=argparse.BooleanOptionalAction)
    parser.add_argument('-i', '--input', type=str, default="target.png", help="Inference input")

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('-d', '--img_dims', type=int, nargs=2, default=[256, 256],
                        help='Height and width')
    parser.add_argument('--hidden_dims', type=int, nargs=2, default=[256, 128])
    parser.add_argument('--colour', action=argparse.BooleanOptionalAction)

    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='Number of canvases painted at once')
    parser.add_argument('-p', '--prims', type=int, default=1,
                        help="Number of primitives output on one run (one input).")
    parser.add_argument('-u', '--updates', type=int, default=1000,
                        help="Number of updates in one batch (update how many times before switching target images).")
    parser.add_argument('-e', '--epochs', type=int, default=1)

    args = parser.parse_args()

    n_channels = 3 if args.colour else 1

    ################################################################################
    # Prepare Environ

    torch.manual_seed(5)

    ################################################################################
    # Prepare Model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    score_fn = LPLoss(2)
    error_fn = LPLoss(2)
    optimise_fn = LPLoss(2)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    ################################################################################
    # Training

    import torchvision

    canvases = None
    one_alphas = torch.ones([args.batch_size, 1, *args.img_dims]).to(device)

    # mse, rel-mse, pnsr, lpips
    error_curve = torch.zeros([args.epochs, args.updates, 4], device=device)

    if args.inference:
        target_imgs = torchvision.io.read_image(
            args.input, torchvision.io.ImageReadMode.RGB if args.colour else torchvision.io.ImageReadMode.GRAY)
        target_imgs = target_imgs.unsqueeze(0)
        target_imgs = target_imgs.to(device)

        canvases, err_curve = inference(target_imgs, args, score_fn, optimise_fn, error_fn, lpips_fn)

        img = torchvision.transforms.ToPILImage()(canvases[0])
        img.save(os.path.join(args.output, "result.png"), format=None)
    else:
        # dataset = ReconDataset(args.img_dims, src="imagenet-r")
        dataset = ImageFolderCustom('testdata')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        for i, target_imgs in enumerate(dataloader):
            if i == args.epochs:
                break

            target_imgs = target_imgs.to(device)

            img = torchvision.transforms.ToPILImage()(target_imgs[0])
            img.save(os.path.join(args.output, "target.png"), format=None)

            canvases, err_curve = inference(target_imgs, args, score_fn, optimise_fn, error_fn, lpips_fn)
            error_curve[i] = err_curve

        torch.save({
            'error_curve': error_curve
        }, os.path.join(args.output, "err_curve.pt"))
