from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import sys



sys.path.append("..")
from utils.train_data import TestData, testDataset_collate, NewTestData, TestImagenet
from config.args import test_options
from config.config import model_config
from compressai.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import ImageFile, Image
from models import *
from utils.testing import test_model
from utils.logger import setup_logger
from utils.newtrain_data import TrainData

def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    torch.backends.cudnn.deterministic = True

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))
    setup_logger('test', os.path.join('./experiments', args.experiment), 'test_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    #
    # test_dataset = TestData("../voc_norm_test.txt")
    # test_dataset = TestImagenet("../../data/liuquan/Imagenet_val25k/val_set.txt")
    # test_dataset = NewTestData("../../data")
    test_dataset = NewTestData("../../data/Kodak24")
    # test_dataset = NewTestData("../../yolov3/coco/val2017")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=testDataset_collate,
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = MLICPlusPlus(config=config)
    net = net.to(device)
    checkpoint = torch.load(args.checkpoint)
    # new_ckpt = modify_checkpoint(checkpoint['state_dict'])
    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint["epoch"]
    logger_test.info(f"Start testing!" )
    save_dir = os.path.join('./experiments', args.experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(net=net, test_dataloader=test_dataloader, logger_test=logger_test, save_dir=save_dir, epoch=epoch)


if __name__ == '__main__':
    main()

