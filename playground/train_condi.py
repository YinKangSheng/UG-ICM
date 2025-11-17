import os
import random
import logging
from PIL import ImageFile, Image
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
import sys



sys.path.append("..")
from models.mlicpp_con import ConditionalMLICPlusPlus
from clip.Closs import CLIPConvLoss
from utils.logger import setup_logger
from utils.newtrain_data import TrainData, dataset_collate
from utils.utils import CustomDataParallel, save_checkpoint
from utils.optimizers import configure_optimizers, configure_optimizer
from utils.training import train_one_epoch, train_clip_one_epoch, train_conditional_one_epoch
from utils.testing import test_one_epoch, test_clip_one_epoch, test_conditional_one_epoch, newtest_conditional_one_epoch
from loss.rd_loss import RateDistortionLoss
from config.args import train_options
from config.config import model_config
from models import *
import random


def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
    #     seed = args.seed
    # else:
        seed = 100 * random.random()
    torch.manual_seed(seed)
    random.seed(seed)

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))

    setup_logger('train', os.path.join('./experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join('./experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('./experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('./experiments', args.experiment, 'checkpoints'))

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = TrainData("../../../coco2voc/coco2voc/ImageSets/Main/trainval.txt")
    test_dataset = TrainData("../../../coco2voc/coco2voc/ImageSets/Main/test.txt")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = ConditionalMLICPlusPlus(config=config)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)
    clip_loss = CLIPConvLoss(device)

    if args.checkpoint != None:
        if args.tune:
            checkpoint = torch.load(args.checkpoint)
            # new_ckpt = modify_checkpoint(checkpoint['state_dict'])
            net.load_state_dict(checkpoint['state_dict'])
            for name, param in net.named_parameters():
                if not ('g_s' in name or 'global_cond' in name):
                    param.requires_grad = False
            # 检查哪些参数被冻结，哪些参数是可训练的
            # for name, param in net.named_parameters():
            #     print(f"{name}: requires_grad={param.requires_grad}")
            optimizer = configure_optimizer(net, args)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
            start_epoch = 0
            best_loss = 1e10
            current_step = 0
        else:
            checkpoint = torch.load(args.checkpoint)
            # new_ckpt = modify_checkpoint(checkpoint['state_dict'])
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450, 550], gamma=0.1)
            # lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
            # lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
            # print(lr_scheduler.state_dict())
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
            current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
            checkpoint = None

    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    # start_epoch = 0
    # best_loss = 1e10
    # current_step = 0

    logger_train.info(args)
    logger_train.info(config)
    logger_train.info(net)
    logger_train.info(optimizer)
    optimizer.param_groups[0]['lr'] = args.learning_rate
    save_dir = os.path.join('./experiments', args.experiment, 'val_images', '%03d' % (1 + 1))
    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_conditional_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            args.tune,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step,
            clip_loss,
            args.lambda_clip,
            args.lmbda,
            args.lambda_beta1,
            args.lambda_beta2
        )


        loss = test_conditional_one_epoch(epoch, test_dataloader, net, criterion, save_dir, logger_val, tb_logger, clip_loss, args.lambda_clip)

        lr_scheduler.step()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        net.update(force=True)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                os.path.join('./experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')

if __name__ == '__main__':
    main()