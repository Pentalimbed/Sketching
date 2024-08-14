import argparse
import os.path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import lpips

from data import ReconDataset, generate_canvases
from raster import SegmentRasteriser, composite_over_alpha, composite_softor, composite_over, get_thickness_px_range

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TrainSketch')
    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-o', '--output', type=str, default="output")

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('-d', '--img_dims', type=int, nargs=2, default=[256, 256],
                        help='Height and width')
    parser.add_argument('--hidden_dims', type=int, nargs=2, default=[256, 128])
    parser.add_argument('--colour', action=argparse.BooleanOptionalAction)

    parser.add_argument('--init_length', type=float, default=0.2)
    parser.add_argument('--straight_through', action=argparse.BooleanOptionalAction)
    parser.add_argument('--perceptual', action=argparse.BooleanOptionalAction)
    parser.add_argument('-l', '--loss_power', type=float, default=1.0)
    parser.add_argument('-t', '--max_thickness', type=float, default=0.2)

    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='Number of canvases painted at once')
    parser.add_argument('-p', '--prims', type=int, default=1,
                        help="Number of primitives output on one run (one input).")
    parser.add_argument('-u', '--updates', type=int, default=1000,
                        help="Number of updates in one batch (update how many times before switching target images).")
    parser.add_argument('-e', '--epochs', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()
    n_channels = 3 if args.colour else 1

    ################################################################################
    # Prepare Environ

    torch.manual_seed(5)

    ################################################################################
    # Prepare Dataset

    dataset = ReconDataset(args.img_dims)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    ################################################################################
    # Prepare Model

    device = 'cuda'

    rasteriser = SegmentRasteriser(args.img_dims, (0.0, args.max_thickness), args.straight_through).to(device)
    compositor = composite_over_alpha
    loss_fn = lpips.LPIPS() if args.perceptual else lambda x, x_target: torch.norm((x - x_target).flatten(1),
                                                                                   p=args.loss_power,
                                                                                   dim=1).sum() / torch.numel(x)
    error_fn = torch.nn.MSELoss()
    optimse_fn = torch.nn.MSELoss()

    ################################################################################
    # Training

    import torchvision

    one_alphas = torch.ones([args.batch_size, 1, *args.img_dims]).to(device)
    pos_encoding = torch.stack(
        torch.meshgrid(torch.linspace(0, 1, args.img_dims[0]), torch.linspace(0, 1, args.img_dims[1]), indexing='ij'))
    pos_encoding = pos_encoding.view(1, *pos_encoding.shape).expand(args.batch_size, *pos_encoding.shape)
    pos_encoding = pos_encoding.to(device)

    error_curve = torch.zeros([args.epochs, args.updates, 2])

    for i, target_imgs in enumerate(dataloader):
        if i == args.epochs:
            break

        target_imgs = target_imgs.to(device)
        canvases = torch.ones([args.batch_size, n_channels, *args.img_dims], device=device)

        initial_error = torch.stack([error_fn(canvases[i], target_imgs[i]) for i in range(args.batch_size)])

        img = torchvision.transforms.ToPILImage()(target_imgs[0])
        img.save(os.path.join(args.output, "target.png"), format=None)

        pbar = trange(args.updates, position=tqdm._get_free_pos())
        for u in pbar:
            canvases = canvases.detach()

            prims = torch.rand([args.batch_size, args.prims, 5], device=device)
            centres = (prims[:, :, :2] + prims[:, :, 2:4]).repeat(1, 1, 2) * 0.5
            prims[:, :, :4] = centres + (prims[:, :, :4] - centres) * args.init_length
            prims.requires_grad_(True)

            colours = torch.rand([args.batch_size, args.prims, n_channels], device=device)
            colours.requires_grad_(True)

            optimizer = torch.optim.Adam([prims, colours], lr=0.02)

            for j in range(100):
                prims.data.clamp_(0, 1)
                prims.grad = None
                optimizer.zero_grad()
                colours.data.clamp_(0, 1)
                colours.grad = None

                layers = rasteriser(torch.cat([prims, torch.ones([*prims.shape[:2], 1], device=device)], dim=2))
                layers = torch.cat([colours.view(*colours.shape[:2], n_channels, 1, 1)
                                   .expand(*colours.shape[:2], n_channels, *layers.shape[-2:]),
                                    layers], dim=2)
                layers = torch.cat([layers,
                                    torch.cat([canvases, one_alphas], dim=1).view(
                                        [args.batch_size, 1, n_channels + 1, *args.img_dims])],
                                   dim=1)

                new_canvases = compositor(layers)

                loss = loss_fn(new_canvases, target_imgs) * args.batch_size
                loss.backward()
                optimizer.step()

                if (u % (args.updates // 10) == 0 or u == args.updates - 1) and (j % 10 == 0 or j == 99):
                    img = torchvision.transforms.ToPILImage()(new_canvases[0])
                    img.save(os.path.join(args.output, f"{u}-move{j}.png"), format=None)

            canvases = new_canvases

            with torch.no_grad():
                error = error_fn(canvases, target_imgs)
                rel_error = (torch.stack([error_fn(canvases[i], target_imgs[i]) for i in range(args.batch_size)]) / initial_error).mean()

            pbar.set_postfix({"error": error.cpu().detach().item(), "rel_error": rel_error.cpu().detach().item()})

            error_curve[i, u, 0] = error
            error_curve[i, u, 0] = rel_error

    torch.save({
        'error_curve': error_curve
    }, os.path.join(args.output, f"err_curve.pt"))
