"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train_data.py
about: build the training dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import gzip
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torch.nn.functional as F

from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding.

    Args:
        in_h: Input height.
        in_w: Input width.
        out_h: Output height.
        out_w: Output width.
        min_div: Length that output dimensions should be divisible by.
    """
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad

def pad(x, p=2**6):
    h, w = x.size(1), x.size(2)
    pad, _ = compute_padding(h, w, min_div=p)
    return F.pad(x, pad, mode="constant", value=0)


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, train_data_dir):
        super().__init__()
        train_list = train_data_dir
        with open(train_list) as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt]

        self.annotations = annotations

    def get_images(self, index):
        seed = torch.random.seed()


        img_name = self.annotations[index]


        image = Image.open("../../../coco2voc/coco2voc/" + 'JPEGImages/' + str(img_name) + '.jpg')
        # --- Transform to tensor --- #
        transform1 = Compose([RandomCrop((256, 256)), ToTensor()])
        transform2 = Compose([Resize((256, 256)), ToTensor()])
        image_width, image_height = image.size
        if image_width >= 256 and image_height >= 256:
            torch.random.manual_seed(seed)
            image = transform1(image)
        else:
            image = transform2(image)

        return image

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.annotations)

class TrainDataWithMask(data.Dataset):
    def __init__(self, train_data_dir):
        super().__init__()
        train_list = train_data_dir
        with open(train_list) as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt]

        self.annotations = annotations

    def get_images(self, index):
        seed = torch.random.seed()

        img_name = self.annotations[index]

        image1 = Image.open("../../../coco2voc/coco2voc/" + 'JPEGImages/' + str(img_name) + '.jpg')
        image2 = image1.copy()
        # --- Transform to tensor --- #
        transform1 = Compose([RandomCrop((256, 256)), ToTensor()])
        transform2 = Compose([Resize((256, 256)), ToTensor()])
        image_width, image_height = image1.size
        if image_width >= 256 and image_height >= 256:
            torch.random.manual_seed(seed)
            image1 = transform1(image1)
        else:
            image1 = transform2(image1)
        image2 = transform2(image2)
        with gzip.open("../../cocomask/cocomask/" + str(img_name) + '.pth.gz', 'rb') as f:
            MaskList = torch.load(f)

        return image1, image2, MaskList



    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.annotations)


# DataLoader中collate_fn使用
def dataset_collate(batch):

    imagesets = []
    for imageset in batch:
        imagesets.append(imageset[0])
    imagesets = torch.from_numpy(np.array([item.numpy() for item in imageset])).type(torch.FloatTensor)
    print(imagesets.shape)
    return imagesets
