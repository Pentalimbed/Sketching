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
            torchvision.transforms.RandomResizedCrop(self.dims, scale=(0.8, 1.0), ratio=(1.0, 1.0))
        ])(image)
        return image


def generate_canvases(target_imgs: torch.Tensor):
    canvases = torch.zeros_like(target_imgs)
    # mean colour
    canvases = target_imgs.flatten(2).mean(dim=2).unsqueeze(-1).unsqueeze(-1).expand(*canvases.shape)

    return canvases
