import os
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
                    datasets.load_dataset('axiong/imagenet-r', cache_dir=r'E:\MLData\datasets\imagenet-r', split='test')
                    .cast_column("image", datasets.Image("RGB"))
                    .select_columns(["image"])
                )
            case "imagenet":
                self.ds = (
                    datasets.load_dataset('ILSVRC/imagenet-1k', cache_dir=r'E:\MLData\datasets\imagenet-1k', split='test',
                                          token="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", trust_remote_code=True)
                    .cast_column("image", datasets.Image("RGB"))
                    .select_columns(["image"])
                )
            case "imagenet":
                self.ds = (
                    datasets.load_dataset('PhilSad/celeba-hq-15k', cache_dir=r'E:\MLData\datasets\celeba-hq', split='test')
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


class ImageFolderCustom(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_paths = os.listdir(img_dir)
        self.transform = transform
    def __getitem__(self, index):
        x = torchvision.io.read_image(os.path.join(self.img_dir, self.image_paths[index]),
                                      torchvision.io.ImageReadMode.RGB)
        x = x / 255.
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)