from typing import Optional

import torch
from torch.utils.data import Dataset
import torchvision
import datasets


class ReconDataset(Dataset):
    def __init__(self, dims, src="imagenet-r"):
        super().__init__()

        self.dims = dims

        match src:
            case "imagenet-r":
                self.ds = (
                    datasets.load_dataset('axiong/imagenet-r', cache_dir='./dataset/imagenet-r', split='test')
                    .cast_column("image", datasets.Image("RGB"))
                    .select_columns(["image"])
                )
            case _:
                raise RuntimeError("Invalid dataset!")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]["image"]
        image = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomResizedCrop(self.dims, scale=(0.5, 1.0), ratio=(1.0, 1.0))
        ])(image)
        return image


def generate_canvases(target_imgs: torch.Tensor, prev_canvases: Optional[torch.Tensor], method_ratio=(0.5, 0.3, 0.2)):
    method_ratio = torch.tensor(method_ratio)
    if prev_canvases is None:
        method_ratio[2] = 0

    n = target_imgs.shape[0]
    breakpoints = torch.cumsum(method_ratio, 0)
    breakpoints = torch.ceil(torch.clamp(breakpoints / breakpoints[-1], 0, 1) * n).type(torch.int)

    canvases = torch.zeros_like(target_imgs)
    # random pure colour
    canvases[:breakpoints[0]] = torch.rand([breakpoints[0], 3, 1, 1]).repeat([1, 1, *target_imgs.shape[2:]])
    # gaussian blur
    canvases[breakpoints[0]:breakpoints[1]] = torchvision.transforms.GaussianBlur(kernel_size=63)(
        target_imgs[breakpoints[0]:breakpoints[1]])
    if prev_canvases is not None:
        # use last
        canvases[breakpoints[1]:] = prev_canvases[breakpoints[1]:].clone()

    return canvases
