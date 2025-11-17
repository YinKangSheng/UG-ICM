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
    def __init__(self, train_data_dir):
        super().__init__()
        train_list = train_data_dir
        with open(train_list) as f:
            txt = f.readlines()
            # annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            annotations = [line.strip() for line in txt]

        self.annotations = annotations
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        line = self.annotations[index].split()
        image_path = line[0]
        # print(image_path)
        img_name = image_path.split('/')[-1]
        # print(img_name)
        image_name = img_name.split('.')[0]
        # print(image_name)
        gt_name = image_name
        pgt_name = image_name

        # --- Transform to tensor --- #
        transform = Compose([ToTensor()])

        gts = []
        haze_img = Image.open("../../yolov3/" + 'VOCdevkit/VOC2007/JPEGImages/' + gt_name +'.jpg' )
        gt = transform(haze_img)
        # --- Check the channel is 3 or not --- #
        if list(gt.shape)[0] is not 3 :
            raise Exception('Bad image channel: {}'.format(gt_name))
        gts.append(gt)

        names = []
        names.append(pgt_name)

        return gts, names

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.annotations)


class TestImagenet(data.Dataset):
    def __init__(self, train_data_dir):
        super().__init__()
        train_list = train_data_dir
        with open(train_list) as f:
            txt = f.readlines()
            # annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            annotations = [line.strip() for line in txt]

        self.annotations = annotations
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        image_path = self.annotations[index]
        # print(image_path)
        img_name = image_path.split('/')[-1]
        # print(img_name)
        image_name = img_name.split('.')[0]
        # print(image_name)
        gt_name = image_name
        pgt_name = image_name

        # --- Transform to tensor --- #
        transform = Compose([ToTensor()])

        gts = []
        haze_img = Image.open("../../data/liuquan/Imagenet_val25k/" + image_path)
        haze_img = haze_img.convert('RGB')
        gt = transform(haze_img)
        # --- Check the channel is 3 or not --- #
        if list(gt.shape)[0] is not 3 :
            raise Exception('Bad image channel: {}'.format(gt_name))
        gts.append(gt)

        names = []
        names.append(pgt_name)

        return gts, names

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.annotations)

class NewTestData(data.Dataset):
    def __init__(self,  img_dir):
        super().__init__()
        self.img_dir = img_dir
        self.img_labels = [f for f in os.listdir(img_dir) if f.endswith(".JPEG") or f.endswith(".png") or f.endswith(".jpg")]

    def get_images(self, index):
        img_label=self.img_labels[index]
        image_name = img_label.split('.')[0]
        img_path = os.path.join(self.img_dir, img_label)

        # --- Transform to tensor --- #
        transform = Compose([ToTensor()])

        gts = []
        haze_img = Image.open(img_path)
        haze_img = haze_img.convert('RGB')
        gt = transform(haze_img)
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


# DataLoader中collate_fn使用
def dataset_collate(batch):
    gts = []
    pgts = []
    for gt, pgt in batch:
        gts.append(gt[0])
        pgts.append(pgt[0])
    gts = torch.from_numpy(np.array([item.numpy() for item in gts])).type(torch.FloatTensor)
    pgts = torch.from_numpy(np.array([item.numpy() for item in pgts])).type(torch.FloatTensor)

    return gts, pgts

def testDataset_collate(batch):
    gts = []
    names = []
    for gt,  name in batch:
        gts.append(gt[0])
        names.append(name[0])
    gts = torch.from_numpy(np.array([item.numpy() for item in gts])).type(torch.FloatTensor)
    return gts,  names