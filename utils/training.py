import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF


def random_crop_batch(batch, crop_size):
    # 获取批次张量的维度
    batch_size, channels, height, width = batch.shape

    # 生成随机裁剪起始位置
    top = torch.randint(0, height - crop_size + 1, (1,)).item()
    left = torch.randint(0, width - crop_size + 1, (1,)).item()

    # 对每个样本进行相同的裁剪
    cropped_batch = torch.stack([TF.crop(image, top, left, crop_size, crop_size) for image in batch])

    return cropped_batch

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
            if out_criterion["mse_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
            if out_criterion["ms_ssim_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

        if i % 100 == 0:
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step


def train_clip_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, clip_loss, lambda_clip
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        closs = clip_loss(out_net["x_hat"], d, "train")
        total_loss = lambda_clip * closs + out_criterion["loss"]
        total_loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), total_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: closs'), closs.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
            if out_criterion["mse_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
            if out_criterion["ms_ssim_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

        if i % 100 == 0:
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step

def train_conditional_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, fine_tune, epoch, clip_max_norm, logger_train, tb_logger, current_step, clip_loss, lambda_clip, lambda_beta0, lambda_beta1, lambda_beta2
):
    model.train()
    device = next(model.parameters()).device
    mse = nn.MSELoss()
    for i, d in enumerate(train_dataloader):
        bs = d.size(0)
        betas = torch.randint(0, 2, (bs,))

        prinst = betas == 0
        image_wise = betas == 1
        # pixel_wise = betas == 2

        prinst_indices = torch.nonzero(prinst).squeeze()
        image_wise_indices = torch.nonzero(image_wise).squeeze()
        # pixel_wise_indices = torch.nonzero(pixel_wise).squeeze()

        d = d.to(device)
        betas = betas.to(device)

        extracted_prinst_d = d[prinst_indices, :, :, :]
        extracted_image_d = d[image_wise_indices, :, :, :]
        # extracted_pixel_d = d[pixel_wise_indices, :, :, :]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model((d, betas))
        out_criterion = criterion(out_net, d)

        # beta=0
        extracted_prinst_x_hat = out_net["x_hat"][prinst_indices, :, :, :]
        extracted_prinst_mse = mse(extracted_prinst_x_hat, extracted_prinst_d)
        # image-wise clip, beta1
        extracted_image_x_hat = out_net["x_hat"][image_wise_indices, :, :, :]
        closs = clip_loss(extracted_image_x_hat, extracted_image_d, "train")
        extracted_image_mse = mse(extracted_image_x_hat, extracted_image_d)


        # pixel-wise clip, beta2


        total_loss = out_criterion["bpp_loss"]+ 0.5 * (lambda_beta0 * 255 ** 2 * extracted_prinst_mse + lambda_beta1 * 255 ** 2 * extracted_image_mse + lambda_clip * closs)
        total_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        if not fine_tune:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), total_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: closs'), closs.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            # tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: prinst_mse_loss'), extracted_prinst_mse.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: image_loss'), extracted_image_mse.item(), current_step)


        if i % 100 == 0:
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    # f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    # f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step



def train_mask_conditional_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, fine_tune, epoch, clip_max_norm, logger_train, tb_logger, current_step, clip_loss, lambda_clip, lambda_beta0, lambda_beta1
):
    model.train()
    device = next(model.parameters()).device
    mse = nn.MSELoss()
    for i, (d1, d2,masklist) in enumerate(train_dataloader):
        bs = d1.size(0)
        betas1 = torch.zeros((bs,))
        betas2 = torch.ones((bs,))

        d1 = d1.to(device)
        d2 = d2.to(device)
        betas1 = betas1.to(device)
        betas2 = betas2.to(device)


        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net1 = model((d1, betas1))
        out_net2 = model((d2, betas2))
        out_criterion1 = criterion(out_net1, d1)


        masked_ori_data = []
        masked_rec_data = []

        for mask in masklist:
            pred_boxs = mask[0]
            pred_masks = mask[1]
            for j, (pred_box, pred_mask) in enumerate(zip(pred_boxs, pred_masks)):
                ori_image = d2[j]
                rec_image = out_net2["x_hat"][j]
                pred_mask, pred_box = pred_mask.type(torch.uint8).to(d2.device), pred_box.type(torch.int).to(d2.device)
                masked_ori_image = ori_image * pred_mask[None, ...]
                masked_rec_image = rec_image * pred_mask[None, ...]

                x1, y1, x2, y2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
                masked_ori_image = TF.resized_crop(masked_ori_image, y1, x1, (y2 - y1), (x2 - x1), (224, 224))
                masked_rec_image = TF.resized_crop(masked_rec_image, y1, x1, (y2 - y1), (x2 - x1), (224, 224))
                masked_ori_data.append(masked_ori_image)
                masked_rec_data.append(masked_rec_image)
        cropped_ori_imgs = torch.stack(masked_ori_data, dim=0)
        cropped_rec_imgs = torch.stack(masked_rec_data, dim=0)

        # beta=0
        extracted_prinst_mse = mse(out_net1["x_hat"], d1)
        #  beta1
        closs = clip_loss(out_net2["x_hat"], d2, "train")
        closs_mask_conv = clip_loss(cropped_rec_imgs, cropped_ori_imgs, "train")


        total_loss = lambda_beta0 * 255 ** 2 * extracted_prinst_mse + lambda_beta1 * closs_mask_conv + lambda_clip * closs
        total_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        if not fine_tune:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), total_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: closs'), closs.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion1["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            # tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: prinst_mse_loss'), extracted_prinst_mse.item(), current_step)
            # tb_logger.add_scalar('{}'.format('[train]: image_loss'), extracted_image_mse.item(), current_step)


        if i % 100 == 0:
            if out_criterion1["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d1):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MSE loss: {out_criterion1["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion1["bpp_loss"].item():.2f} | '
                    # f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d1):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MS-SSIM loss: {out_criterion1["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion1["bpp_loss"].item():.2f} | '
                    # f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step

def train_mutil_conditional_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, fine_tune, epoch, clip_max_norm, logger_train, tb_logger, current_step, clip_loss, lambda_local, ambda_instance, lambda_global, lmbda
):
    model.train()
    device = next(model.parameters()).device
    mse = nn.MSELoss()
    for i, (d1, d2, masklist) in enumerate(train_dataloader):
        bs = d1.size(0)
        betas1 = torch.zeros((bs,))
        betas2 = torch.ones((bs,))

        d1 = d1.to(device)
        d2 = d2.to(device)
        betas1 = betas1.to(device)
        betas2 = betas2.to(device)


        optimizer.zero_grad()
        aux_optimizer.zero_grad()


        out_net1 = model((d1, betas1))
        out_net2 = model((d2, betas2))
        out_criterion1 = criterion(out_net1, d1)


        masked_ori_data = []
        masked_rec_data = []

        for mask in masklist:
            pred_boxs = mask[0]
            pred_masks = mask[1]
            for j, (pred_box, pred_mask) in enumerate(zip(pred_boxs, pred_masks)):
                ori_image = d2[j]
                rec_image = out_net2["x_hat"][j]
                pred_mask, pred_box = pred_mask.type(torch.uint8).to(d2.device), pred_box.type(torch.int).to(d2.device)
                masked_ori_image = ori_image * pred_mask[None, ...]
                masked_rec_image = rec_image * pred_mask[None, ...]

                x1, y1, x2, y2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
                masked_ori_image = TF.resized_crop(masked_ori_image, y1, x1, (y2 - y1), (x2 - x1), (224, 224))
                masked_rec_image = TF.resized_crop(masked_rec_image, y1, x1, (y2 - y1), (x2 - x1), (224, 224))
                masked_ori_data.append(masked_ori_image)
                masked_rec_data.append(masked_rec_image)
        cropped_ori_imgs = torch.stack(masked_ori_data, dim=0)
        cropped_rec_imgs = torch.stack(masked_rec_data, dim=0)

        merged_tensor = torch.cat((out_net2["x_hat"], d2), dim=0)
        output_batch = random_crop_batch(merged_tensor, 224)
        local_rec_imgs, local_ori_imgs = torch.split(output_batch, bs)
        # beta=0
        extracted_prinst_mse = mse(out_net1["x_hat"], d1)
        #  beta1
        closs = clip_loss(out_net2["x_hat"], d2, "train")
        closs_local = clip_loss(local_rec_imgs, local_ori_imgs, "train")
        closs_mask_conv = clip_loss(cropped_rec_imgs, cropped_ori_imgs, "train")


        total_loss = lmbda * 255 ** 2 * extracted_prinst_mse + ambda_instance * closs_mask_conv + lambda_global * closs + lambda_local *  closs_local
        total_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        if not fine_tune:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), total_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: closs'), closs.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion1["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            # tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: prinst_mse_loss'), extracted_prinst_mse.item(), current_step)
            # tb_logger.add_scalar('{}'.format('[train]: image_loss'), extracted_image_mse.item(), current_step)


        if i % 100 == 0:
            if out_criterion1["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d1):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MSE loss: {out_criterion1["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion1["bpp_loss"].item():.2f} | '
                    # f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d1):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MS-SSIM loss: {out_criterion1["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion1["bpp_loss"].item():.2f} | '
                    # f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step


def warmup_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, lr_scheduler
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        if epoch < 1:
            lr_scheduler.step()
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
            if out_criterion["mse_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
            if out_criterion["ms_ssim_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

        if i % 100 == 0:
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Lr: {optimizer.param_groups[0]["lr"]:.6f} | '
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Lr: {optimizer.param_groups[0]["lr"]:.6f} | '
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step