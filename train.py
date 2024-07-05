import argparse
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from data import ReconDataset, generate_canvases
from models import StrokePredictor
from raster import SegmentRasteriser, composite_over, composite_over_alpha

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TrainSketch')
    parser.add_argument('-c', '--checkpoint')

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('-d', '--img_dims', type=int, nargs=2, default=[256, 256],
                        help='Height and width')
    parser.add_argument('--latent_dims', type=int, default=1024)
    parser.add_argument('--hidden_dims', type=int, nargs=2, default=[64, 256])

    parser.add_argument('-b', '--batch_size', type=int, default=10,
                        help='Number of canvases painted at once')
    parser.add_argument('-p', '--prims', type=int, default=1,
                        help="Number of primitives output on one run (one input).")
    parser.add_argument('-r', '--runs', type=int, default=1,
                        help="Number of runs in one update (draw how many times before backprop and update).")
    parser.add_argument('-u', '--updates', type=int, default=100,
                        help="Number of updates in one batch (update how many times before switching target images).")
    parser.add_argument('-e', '--epochs', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    ################################################################################
    # Prepare Environ

    torch.manual_seed(0)

    ################################################################################
    # Prepare Dataset

    dataset = ReconDataset(args.img_dims)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    ################################################################################
    # Prepare Model

    device = torch.cuda.device(0) if torch.cuda.is_available() else 'cpu'

    model = StrokePredictor(latent_dim=args.latent_dims, hidden_dims=args.hidden_dims, in_channels=3 * 2,
                            prim_num=args.prims).to(device)
    rasteriser = SegmentRasteriser(args.img_dims).to(device)
    compositor = composite_over_alpha
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    canvases = None
    one_alphas = torch.ones([args.batch_size, 1, *args.img_dims]).to(device)

    import matplotlib.pyplot as plt
    import torchvision

    for e in range(args.epochs - epochs):
        for target_imgs in dataloader:
            canvases = generate_canvases(target_imgs, canvases)

            plt.figure()
            plt.imshow(torchvision.transforms.ToPILImage()(canvases[0]))
            plt.waitforbuttonpress(1)

            plt.figure()
            plt.imshow(torchvision.transforms.ToPILImage()(target_imgs[0]))
            plt.waitforbuttonpress(1)

            pbar = trange(args.updates, desc=f"Epoch {e}:")
            for u in pbar:
                optimiser.zero_grad()
                canvases = canvases.detach()

                prev_loss = torch.nn.MSELoss()(canvases, target_imgs)

                for r in range(args.runs):
                    x = torch.cat([target_imgs, canvases], dim=1).to(device)
                    prims = model(x)
                    layers = rasteriser(prims)
                    layers = torch.cat(
                        [layers,
                         torch.cat([canvases, one_alphas], dim=1).view(
                             [args.batch_size, 1, 4, *args.img_dims])],
                        dim=1)
                    canvases = compositor(layers)

                loss = torch.nn.MSELoss()(canvases, target_imgs)
                loss = loss - prev_loss
                loss.backward()
                optimiser.step()

                pbar.set_postfix({"Loss": loss.detach().item()})

            plt.figure()
            plt.imshow(torchvision.transforms.ToPILImage()(canvases[0]))
            plt.waitforbuttonpress(1)