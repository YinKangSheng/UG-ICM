"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train_data.py
about: build the training dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import os
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


class TestData(data.Dataset):
    def __init__(self,  img_dir):
        super().__init__()
        self.img_dir = img_dir
        self.img_labels = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]

    def get_images(self, index):
        img_label=self.img_labels[index]
        image_name = img_label.split('.')[0]
        img_path = os.path.join(self.img_dir, img_label)

        # --- Transform to tensor --- #
        transform = Compose([ToTensor()])

        gts = []
        img = Image.open(img_path)
        img = img.convert('RGB')
        gt = transform(img)
        # --- Check the channel is 3 or not --- #
        if list(gt.shape)[0] is not 3 :
            raise Exception('Bad image channel: {}'.format(image_name))
        gts.append(gt)

        names = []
        names.append(image_name)

        return gts, names

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.img_labels)



def testDataset_collate(batch):
    gts = []
    names = []
    for gt,  name in batch:
        gts.append(gt[0])
        names.append(name[0])
    gts = torch.from_numpy(np.array([item.numpy() for item in gts])).type(torch.FloatTensor)
    return gts,  names