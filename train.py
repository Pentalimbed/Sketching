import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import lpips

from data import ReconDataset, generate_canvases
from models import StrokePredictor
from raster import SegmentRasteriser, composite_over_alpha, composite_softor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TrainSketch')
    parser.add_argument('-c', '--checkpoint')

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('-d', '--img_dims', type=int, nargs=2, default=[256, 256],
                        help='Height and width')
    parser.add_argument('--hidden_dims', type=int, nargs=2, default=[256, 128])
    parser.add_argument('--colour', action=argparse.BooleanOptionalAction)
    parser.add_argument('--thickness', action=argparse.BooleanOptionalAction)

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
    rasteriser = SegmentRasteriser(args.img_dims, straight_through=True).to(device)
    compositor = composite_over_alpha
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    # loss_fn = torch.nn.SmoothL1Loss()
    # loss_fn = lambda x, x_target: torch.pow(torch.norm((x - x_target).flatten(1), p=4, dim=1).sum() / torch.numel(x),
    #                                         1 / 4.0)

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
        prims = torch.rand([1, args.prims, 5 + n_channels], device=device)
        centres = (prims[:, :, :2] + prims[:, :, 2:4]).repeat(1, 1, 2) * 0.5
        prims[:, :, :4] = centres + (prims[:, :, :4] - centres) * 0.2
        prims.requires_grad_(True)
        prims_optim = torch.optim.Adam([prims], lr=0.01)

        for j in range(100):
            # with torch.no_grad():
            #     centres = (prims[:, :, :2] + prims[:, :, 2:4]).repeat(1, 1, 2) * 0.5
            #     lengths = torch.norm(prims[:, :, :2] - prims[:, :, 2:4], dim=2, keepdim=True) + torch.finfo().eps
            #     prims[:, :, :4] = centres + (prims[:, :, :4] - centres) * torch.minimum(lengths, length_limit) / lengths

            prims.data.clamp_(0, 1)
            prims.grad = None
            prims_optim.zero_grad()

            layers = rasteriser(
                torch.cat([prims[:, :, :5], torch.ones([*prims.shape[:2], 1], device=device)], dim=2))
            layers = torch.cat([prims[:, :, 5:]
                               .view(*prims.shape[:2], n_channels, 1, 1)
                               .expand(*prims.shape[:2], n_channels, *layers.shape[-2:]),
                                layers], dim=2)
            layers = torch.cat(
                [layers,
                 torch.cat([canvases, one_alphas], dim=1).view([args.batch_size, 1, n_channels + 1, *args.img_dims])],
                dim=1)

            new_canvases = compositor(layers)

            loss = loss_fn(new_canvases, target.unsqueeze(0))
            loss.backward()
            prims_optim.step()

        canvases = new_canvases.detach()
        pbar.set_postfix({"Loss": loss.detach().cpu().item()})

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
