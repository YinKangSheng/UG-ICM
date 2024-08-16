from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import sys



sys.path.append("..")
from utils.utils import save_checkpoint
from utils.train_data import TestData, testDataset_collate

from config.args import test_options
from config.config import model_config

from torch.utils.data import DataLoader
from PIL import ImageFile, Image
from models import *
from utils.testing import test_model
from utils.logger import setup_logger


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


    test_dataset = TestData(args.dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=testDataset_collate,
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = UG_MLICPlusPlus(config=config)
    net = net.to(device)
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    logger_test.info(f"Start testing!" )
    save_dir = os.path.join('./experiments', args.experiment, "bitstream")
    save_dir_human = os.path.join('./experiments', args.experiment, "human")
    save_dir_machine = os.path.join('./experiments', args.experiment, "machine")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_human):
        os.makedirs(save_dir_human)
    if not os.path.exists(save_dir_machine):
        os.makedirs(save_dir_machine)
    test_model(net=net, test_dataloader=test_dataloader, logger_test=logger_test, save_dir=save_dir ,save_dir_human=save_dir_human ,save_dir_machine=save_dir_machine)


if __name__ == '__main__':
    main()

