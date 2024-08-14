import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import lpips

from data import ReconDataset, generate_canvases
from models import StrokePredictor
from raster import SegmentRasteriser, composite_over_alpha, composite_softor, composite_over

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TrainSketch')
    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-o', '--output', type=str, default="output")

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('-d', '--img_dims', type=int, nargs=2, default=[256, 256],
                        help='Height and width')
    parser.add_argument('--hidden_dims', type=int, nargs=2, default=[256, 128])
    parser.add_argument('--colour', action=argparse.BooleanOptionalAction)

    parser.add_argument('--perceptual', action=argparse.BooleanOptionalAction)
    parser.add_argument('-l', '--loss_power', type=float, default=1.0)
    parser.add_argument('-t', '--max_thickness', type=float, default=1.0)

    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Number of canvases painted at once')
    parser.add_argument('-p', '--prims', type=int, default=1,
                        help="Number of primitives output on one run (one input).")
    parser.add_argument('-r', '--runs', type=int, default=1,
                        help="Number of runs in one update (draw how many times before backprop and update).")
    parser.add_argument('-u', '--updates', type=int, default=800,
                        help="Number of updates in one batch (update how many times before switching target images).")
    parser.add_argument('-e', '--epochs', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.001)

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

    model = StrokePredictor(hidden_dims=args.hidden_dims, in_channels=n_channels * 2 + 2, prim_num=args.prims).to(
        device)
    rasteriser = SegmentRasteriser(args.img_dims, straight_through=False).to(device)
    compositor = composite_over_alpha
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    prims_loss_fn = torch.nn.SmoothL1Loss()
    # colour_loss_fn = lambda x, x_target: torch.norm((x - x_target).flatten(1), p=0.5, dim=1).sum() / torch.numel(x)
    colour_loss_fn = torch.nn.SmoothL1Loss()

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

    target = dataset[192].to(device)
    if not args.colour:
        target = torchvision.transforms.Grayscale()(target)

    img = torchvision.transforms.ToPILImage()(target)
    img.save(f"outputs/target.png", format=None)

    # length_limit = torch.tensor(0.25, device=device)

    canvases = torch.ones([1, n_channels, *args.img_dims], device=device)
    pbar = trange(args.updates)
    for i in pbar:
        prims = torch.rand([1, args.prims, 5], device=device)
        centres = (prims[:, :, :2] + prims[:, :, 2:4]).repeat(1, 1, 2) * 0.5
        prims[:, :, :4] = centres + (prims[:, :, :4] - centres) * 0.2
        prims.requires_grad_(True)

        colours = torch.rand([1, args.prims, n_channels], device=device)
        colours.requires_grad_(True)

        prims_optim = torch.optim.Adam([prims, colours], lr=0.01)
        # colour_optim = torch.optim.Adam([colours], lr=0.1)

        # completeness = torch.nn.L1Loss()(canvases, target.unsqueeze(0))

        for j in range(200):
            # with torch.no_grad():
            #     centres = (prims[:, :, :2] + prims[:, :, 2:4]).repeat(1, 1, 2) * 0.5
            #     lengths = torch.norm(prims[:, :, :2] - prims[:, :, 2:4], dim=2, keepdim=True) + torch.finfo().eps
            #     prims[:, :, :4] = centres + (prims[:, :, :4] - centres) * torch.minimum(lengths, length_limit) / lengths

            prims.data.clamp_(0, 1)
            with torch.no_grad():
                prims[:, :, 4] = prims[:, :, 4].clamp_max(
                    torch.lerp(torch.tensor(1.0, device=device), torch.tensor(0.01, device=device),
                               torch.pow(torch.tensor(i / (args.updates - 1), device=device), 0.5)))
            prims.grad = None
            prims_optim.zero_grad()

            colours.data.clamp_(0, 1)
            colours.grad = None
            # colour_optim.zero_grad()

            layers = rasteriser(
                torch.cat([prims, torch.ones([*prims.shape[:2], 1], device=device)], dim=2))

            # influences = torch.cat([layers, torch.zeros([args.batch_size, 1, 1, *args.img_dims], device=device)], dim=1)
            # influences = composite_over(influences)

            layers = torch.cat([colours.view(*colours.shape[:2], n_channels, 1, 1)
                               .expand(*colours.shape[:2], n_channels, *layers.shape[-2:]),
                                layers], dim=2)
            layers = torch.cat(
                [layers,
                 torch.cat([canvases, one_alphas], dim=1).view([args.batch_size, 1, n_channels + 1, *args.img_dims])],
                dim=1)

            new_canvases = compositor(layers)

            err = target.unsqueeze(0) - new_canvases

            # colour_loss = prims_loss_fn(err, torch.zeros_like(err))
            # loss = colour_loss.clone()

            # oppo_cost = torch.nn.L1Loss()(err * (1 - influences), torch.zeros_like(err))
            # change = torch.nn.L1Loss()(new_canvases, canvases).detach().item()
            # small_change_penalty = (1.0 - min(1.0, change / (completeness * 0.01))) * oppo_cost * 0.5
            # loss = colour_loss + small_change_penalty

            loss = prims_loss_fn(new_canvases, target.unsqueeze(0))
            # colour_loss = colour_loss_fn(new_canvases, target.unsqueeze(0))

            # colours.requires_grad_(False)
            loss.backward(retain_graph=False)
            prims_optim.step()

            # colours.requires_grad_(True)
            # prims.requires_grad_(False)
            # colour_loss.backward()
            # colour_optim.step()
            # prims.requires_grad_(True)

            # if i in [0, 20, 100, 200, 400, 799] and (j % 20 == 0 or j == 199):
            #     img = torchvision.transforms.ToPILImage()(new_canvases[0])
            #     img.save(f"outputs/{i}-move{j}.png", format=None)

        canvases = new_canvases.detach()
        pbar.set_postfix({"loss": loss.detach().cpu().item()})

        if i % (args.updates // 10) == 0 or i == args.updates - 1:
            img = torchvision.transforms.ToPILImage()(new_canvases[0])
            img.save(f"outputs/{i}.png", format=None)

    # for e in range(args.epochs - epochs):
    #     pbar = tqdm(dataloader, desc=f"Epoch {e}:", position=0)
    #     for target_imgs in pbar:
    #         display_idx = torch.randint(0, args.batch_size, [1]).flatten().item()
    #
    #         canvases = generate_canvases(target_imgs)
    #         display_old_canvas = canvases[display_idx]
    #
    #         target_imgs = target_imgs.to(device)
    #         canvases = canvases.to(device)
    #
    #         for u in range(args.updates):
    #             optimiser.zero_grad()
    #             canvases = canvases.detach()
    #
    #             for r in range(args.runs):
    #                 x = torch.cat([target_imgs, canvases, pos_encoding], dim=1)
    #                 prims = model(x)
    #                 layers = rasteriser(
    #                     torch.cat([prims[:, :, :4], torch.ones([*prims.shape[:2], 2], device=device)], dim=2))
    #                 layers = torch.cat([
    #                     prims[:, :, 4:].view(*prims.shape[:2], n_channels, 1, 1).expand(*prims.shape[:2], n_channels, *layers.shape[-2:]),
    #                     layers], dim=2)
    #                 layers = torch.cat(
    #                     [layers,
    #                      torch.cat([canvases, one_alphas], dim=1).view([args.batch_size, 1, n_channels + 1, *args.img_dims])],
    #                     dim=1)
    #
    #                 canvases = compositor(layers)
    #
    #             loss = loss_fn(canvases * 2 - 1, target_imgs * 2 - 1).mean()
    #             if torch.isnan(loss).any():
    #                 raise RuntimeError("Help NAN!")
    #
    #             loss.backward()
    #             optimiser.step()
    #
    #             pbar.set_postfix({"Loss": loss.cpu().detach().item()})
    #
    #         fig, axs = plt.subplots(1, 3)
    #         axs[0].imshow(torchvision.transforms.ToPILImage()(display_old_canvas))
    #         axs[1].imshow(torchvision.transforms.ToPILImage()(canvases[display_idx].cpu()))
    #         axs[2].imshow(torchvision.transforms.ToPILImage()(target_imgs[display_idx].cpu()))
    #         fig.tight_layout()
    #         fig.savefig("outputs/compare.png")
    #         plt.close(fig)
    #
    #         torch.save({
    #             'epoch': e,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimiser.state_dict(),
    #             'loss': loss,
    #         }, f"outputs/{e}.pt")
