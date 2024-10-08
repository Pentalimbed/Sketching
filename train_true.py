import argparse
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import lpips

from data import ReconDataset, generate_canvases
from models import StrokePredictor
from raster import SegmentRasteriser, composite_over_alpha, composite_softor, composite_over, get_thickness_px_range


def remap(value, from_range, to_range):
    return (torch.lerp(from_range[0], to_range[1], value) - to_range[0]) / (to_range[1] - to_range[0])


def raw_optimise(
        canvases: torch.Tensor, target: torch.Tensor,
        initial_prims: torch.Tensor, initial_colours: torch.Tensor,
        score_fn: Callable = torch.nn.L1Loss(), prims_loss_fn: Callable = torch.nn.MSELoss(), colours_loss_fn=None,
        steps=100, enable_prims=True, enable_colour=True, sigma=0.2,
        thickness_range=(0.0, 1.0)):
    batch_size = canvases.shape[0]
    n_channels = canvases.shape[1]
    img_dims = canvases.shape[2:]
    device = canvases.device

    in_range = get_thickness_px_range(0.0, 1.0, img_dims, device)
    out_range = get_thickness_px_range(*thickness_range, img_dims, device)

    rasteriser = SegmentRasteriser(img_dims, thickness_range, straight_through=True)
    rasteriser.to(device)

    one_alphas = torch.ones([batch_size, 1, *img_dims]).to(device)

    prims = initial_prims + torch.normal(0, sigma, initial_prims.size(), device=device)
    prims[:, :, 4] = remap(prims[:, :, 4], in_range, out_range)
    prims = prims.clone().detach()
    prims.requires_grad_(enable_prims)

    colours = (initial_colours + torch.normal(0, sigma, initial_colours.size(), device=device)).clone().detach()
    colours.requires_grad_(enable_colour)

    if enable_prims:
        prims_optim = torch.optim.Adam([prims], lr=0.02)
    if enable_colour:
        colour_optim = torch.optim.Adam([colours], lr=0.1)

    for j in range(steps):
        if enable_prims:
            prims.data.clamp_(0, 1)
            prims.grad = None
            prims_optim.zero_grad()

        if enable_colour:
            colours.data.clamp_(0, 1)
            colours.grad = None
            colour_optim.zero_grad()

        layers = rasteriser(torch.cat([prims, torch.ones([*prims.shape[:2], 1], device=device)], dim=2))
        layers = torch.cat([colours.view(*colours.shape[:2], n_channels, 1, 1)
                           .expand(*colours.shape[:2], n_channels, *layers.shape[-2:]),
                            layers], dim=2)
        layers = torch.cat([layers,
                            torch.cat([canvases, one_alphas], dim=1).view([batch_size, 1, n_channels + 1, *img_dims])],
                           dim=1)

        new_canvases = compositor(layers)

        loss = prims_loss_fn(new_canvases, target) * batch_size
        if colours_loss_fn is not None:
            colour_loss = colours_loss_fn(new_canvases, target) * batch_size
        else:
            colour_loss = loss.clone()

        if enable_prims:
            if enable_colour:
                colours.requires_grad_(True)
            loss.backward(retain_graph=enable_colour)
            prims_optim.step()

        if enable_colour:
            if enable_prims:
                prims.requires_grad_(False)

            colours.requires_grad_(True)
            colour_loss.backward()
            colour_optim.step()

            if enable_prims:
                prims.requires_grad_(True)

    with torch.no_grad():
        scores = torch.tensor([score_fn(target[i], new_canvases[i]) for i in range(batch_size)], device=device)
        # rescale thickness range
        prims[:, :, 4] = remap(prims[:, :, 4], out_range, in_range)
    return prims.detach(), colours.detach(), scores, new_canvases.detach()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TrainSketch')
    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-o', '--output', type=str, default="output")

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('-d', '--img_dims', type=int, nargs=2, default=[256, 256],
                        help='Height and width')
    parser.add_argument('--hidden_dims', type=int, nargs=2, default=[256, 128])
    parser.add_argument('--colour', action=argparse.BooleanOptionalAction)

    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Number of canvases painted at once')
    parser.add_argument('-p', '--prims', type=int, default=1,
                        help="Number of primitives output on one run (one input).")
    parser.add_argument('-u', '--updates', type=int, default=800,
                        help="Number of updates in one batch (update how many times before switching target images).")
    parser.add_argument('-e', '--epochs', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()
    n_channels = 3 if args.colour else 1

    ################################################################################
    # Prepare Environ

    torch.manual_seed(166)

    ################################################################################
    # Prepare Dataset

    dataset = ReconDataset(args.img_dims)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    ################################################################################
    # Prepare Model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    model = StrokePredictor(
        hidden_dims=args.hidden_dims,
        in_channels=n_channels * 2 + 2,
        prim_num=args.prims).to(device)
    rasteriser = SegmentRasteriser(args.img_dims, straight_through=True).to(device)
    compositor = composite_over_alpha
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    score_fn = lambda x, x_target: torch.norm((x - x_target).flatten(1), p=0.5, dim=1).sum() / torch.numel(x)
    loss_fn = torch.nn.MSELoss()
    completeness_fn = lambda x, x_target: ((torch.abs(x - x_target).flatten(1) ** 2).sum()) / torch.numel(x)
    optimse_fn = torch.nn.MSELoss()

    epochs = 0
    loss = torch.tensor(0)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['epochs']
        loss = checkpoint['loss']

    loss = loss.to(device)

    ################################################################################
    # Training

    import torchvision

    canvases = None
    one_alphas = torch.ones([args.batch_size, 1, *args.img_dims]).to(device)
    pos_encoding = torch.stack(
        torch.meshgrid(torch.linspace(0, 1, args.img_dims[0]), torch.linspace(0, 1, args.img_dims[1]), indexing='ij'))
    pos_encoding = pos_encoding.view(1, *pos_encoding.shape).expand(args.batch_size, *pos_encoding.shape)
    pos_encoding = pos_encoding.to(device)

    loss_curve = []
    completeness_curve = []

    for e in range(args.epochs - epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {e}:", position=0)
        for target_imgs in pbar:
            display_idx = torch.randint(0, args.batch_size, [1]).flatten().item()

            canvases = generate_canvases(target_imgs)
            display_old_canvas = canvases[display_idx]

            # target_imgs = target_imgs.to(device)
            target_imgs = dataset[2648].unsqueeze(0).to(device)
            canvases = canvases.to(device)

            img = torchvision.transforms.ToPILImage()(target_imgs[0])
            img.save(f"outputs/target.png", format=None)

            for u in range(args.updates):
                optimiser.zero_grad()
                canvases = canvases.detach()

                # x = torch.cat([target_imgs, canvases, pos_encoding], dim=1)
                # prims, colours = model(x)
                # prims = torch.rand([args.batch_size, args.prims, 5], device=device)
                # colours = torch.rand([args.batch_size, args.prims, n_channels], device=device)

                # with torch.no_grad():
                #     layers = rasteriser(torch.cat([prims, torch.ones([*prims.shape[:2], 1], device=device)], dim=2))
                #     layers = torch.cat([
                #         colours.view(*colours.shape[:2], n_channels, 1, 1)
                #         .expand(*colours.shape[:2], n_channels, *layers.shape[-2:]),
                #         layers], dim=2)
                #     layers = torch.cat([layers,
                #                         torch.cat([canvases, one_alphas], dim=1)
                #                        .view([args.batch_size, 1, n_channels + 1, *args.img_dims])], dim=1)
                #
                #     new_canvases = compositor(layers)
                #     scores = torch.tensor([score_fn(new_canvases[i], target_imgs[i]) for i in range(args.batch_size)],
                #                           device=device)

                candidates = [
                    raw_optimise(canvases, target_imgs, torch.rand([args.batch_size, args.prims, 5], device=device),
                                 torch.rand([args.batch_size, args.prims, n_channels], device=device),
                                 prims_loss_fn=optimse_fn, score_fn=score_fn,
                                 thickness_range=(0, 1.0), sigma=0, steps=100),
                    raw_optimise(canvases, target_imgs, torch.rand([args.batch_size, args.prims, 5], device=device),
                                 torch.rand([args.batch_size, args.prims, n_channels], device=device),
                                 prims_loss_fn=optimse_fn, score_fn=score_fn,
                                 thickness_range=(0, 0.2), sigma=0, steps=100),
                    raw_optimise(canvases, target_imgs, torch.rand([args.batch_size, args.prims, 5], device=device),
                                 torch.rand([args.batch_size, args.prims, n_channels], device=device),
                                 prims_loss_fn=optimse_fn, score_fn=score_fn,
                                 thickness_range=(0, 0.1), sigma=0, steps=100),
                    raw_optimise(canvases, target_imgs, torch.rand([args.batch_size, args.prims, 5], device=device),
                                 torch.rand([args.batch_size, args.prims, n_channels], device=device),
                                 prims_loss_fn=optimse_fn, score_fn=score_fn,
                                 thickness_range=(0, 0.02), sigma=0, steps=100),
                    # (prims.detach(), colours.detach(), scores.detach(), new_canvases.detach())
                ]

                all_scores = torch.stack([candidate[2] for candidate in candidates])
                best_candidate = torch.argmin(all_scores, dim=0)

                target_prims = torch.stack(
                    [candidates[best_candidate[i]][0][i] for i in range(best_candidate.shape[0])])
                target_colours = torch.stack(
                    [candidates[best_candidate[i]][1][i] for i in range(best_candidate.shape[0])])
                canvases = torch.stack([candidates[best_candidate[i]][3][i] for i in range(best_candidate.shape[0])])

                # loss = loss_fn(torch.cat([prims, colours], dim=2), torch.cat([target_prims, target_colours], dim=2))
                # if torch.isnan(loss).any():
                #     raise RuntimeError("Help NAN!")
                # loss.backward()
                # optimiser.step()

                with torch.no_grad():
                    loss = loss_fn(canvases, target_imgs)
                    completeness = completeness_fn(canvases, target_imgs)

                pbar.set_postfix({"Progress": f"{u}/{args.updates}",
                                  "Loss": loss.cpu().detach().item(),
                                  "Completeness": completeness.cpu().detach().item()})

                # fig, axs = plt.subplots(1, 3)
                # axs[0].imshow(torchvision.transforms.ToPILImage()(display_old_canvas))
                # axs[1].imshow(torchvision.transforms.ToPILImage()(canvases[display_idx].cpu()))
                # axs[2].imshow(torchvision.transforms.ToPILImage()(target_imgs[display_idx].cpu()))
                # fig.tight_layout()
                # fig.savefig("outputs/compare.png")
                # plt.close(fig)

                if u % (args.updates // 10) == 0 or u == args.updates - 1:
                    img = torchvision.transforms.ToPILImage()(canvases[0])
                    img.save(f"outputs/{u}.png", format=None)

            break

            loss_curve.append(loss)
            completeness_curve.append(completeness)

            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': loss,
                'loss_curve': torch.tensor(loss_curve),
                'completeness_curve': torch.tensor(completeness_curve)
            }, f"outputs/{e}.pt")
