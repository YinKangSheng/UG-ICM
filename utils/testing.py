import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import compute_metrics, compute_metrics2
from utils.utils import *



def compress_one_image(model, x, stream_path, H, W, img_name):
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    return bpp, out["cost_time"]


def decompress_one_image(model, stream_path, img_name, beta):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape, beta)

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
    return x_hat, cost_time



def test_model(test_dataloader, net, logger_test, save_dir, save_dir_human, save_dir_machine):
    net.eval()
    device = next(net.parameters()).device
    avg_enc_time = AverageMeter()
    avg_bpp= AverageMeter()

    avg_psnr_human = AverageMeter()
    avg_ssim_human = AverageMeter()
    avg_dec_time_human = AverageMeter()

    avg_psnr_machine = AverageMeter()
    avg_ssim_machine = AverageMeter()
    avg_dec_time_machine = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            img,  names = data
            img = img.to(device)
            B, C, H, W = img.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            # warmup GPU
            if i == 0:
                bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            # avoid resolution leakage
            net.update_resolutions(16, 16)
            bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            # avoid resolution leakage
            net.update_resolutions(16, 16)
            avg_bpp.update(bpp)
            avg_enc_time.update(enc_time)

            x_hat_human, dec_time = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i), beta=0)
            rec_human = torch2img(x_hat_human)
            img = torch2img(img)
            rec_human.save(os.path.join(save_dir_human, names[0]) + ".png")
            p, m = compute_metrics2(rec_human, img)
            avg_psnr_human.update(p)
            avg_ssim_human.update(m)
            avg_dec_time_human.update(dec_time)
            logger_test.info(
                f"For Human | "
                f"Image[{i}] | "
                f"Bpp loss: {bpp:.2f} | "
                f"PSNR: {p:.4f} | "
                f"SSIM: {m:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding Latency: {dec_time:.4f}"
            )
            x_hat_machine, dec_time2 = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i), beta=1)
            rec_machine = torch2img(x_hat_machine)
            rec_machine.save(os.path.join(save_dir_machine, names[0]) + ".png")
            p2, m2 = compute_metrics2(rec_machine, img)
            avg_psnr_machine.update(p2)
            avg_ssim_machine.update(m2)
            avg_dec_time_machine.update(dec_time2)
            logger_test.info(
                f"For Machine | "
                f"Image[{i}] | "
                f"Bpp loss: {bpp:.2f} | "
                f"PSNR: {p2:.4f} | "
                f"SSIM: {m2:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding Latency: {dec_time2:.4f}"
            )
    logger_test.info(
        f"For Human | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr_human.avg:.4f} | "
        f"Avg SSIM: {avg_ssim_human.avg:.4f} | "
        f"Avg Encoding Latency:: {avg_enc_time.avg:.4f} | "
        f"Avg decoding Latency:: {avg_dec_time_human.avg:.4f}"
    )
    logger_test.info(
        f"For Machine | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr_machine.avg:.4f} | "
        f"Avg SSIM: {avg_ssim_machine.avg:.4f} | "
        f"Avg Encoding Latency:: {avg_enc_time.avg:.4f} | "
        f"Avg decoding Latency:: {avg_dec_time_machine.avg:.4f}"
    )



