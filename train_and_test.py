import sys
import torch
import numpy as np
import time
from PatchUp.loss import GradientLoss
from pre_post_processing import post_processing

def train_one_epoch(epoch, epochs, device, model, dl, optim, loss, prints=False):
    loss_list = []
    model.train()


    for i, batch in enumerate(dl):
        optim.zero_grad(set_to_none=True)

        input, target = batch['img'].to(device), batch['label'].to(device)
        # input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=0.0)
        #
        # if not torch.isfinite(input).all():
        #     print("NaN in input")

        output = model(input)
        # if not torch.isfinite(output).all():
        #     print("NaN in output")

        batch_loss = loss(output, target)
        # if not torch.isfinite(batch_loss):
        #     print("NaN loss at batch", i)
        #     break

        loss_list.append(batch_loss.item())

        batch_loss.backward()
        optim.step()

        if prints:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Batch loss: %f] [LR: %f]"
                % (
                    epoch + 1,
                    epochs,
                    i + 1,
                    len(dl),
                    batch_loss.item(),
                    optim.param_groups[0]['lr'],
                )
            )

    total_loss = np.mean(loss_list)
    return total_loss

def validate(device, model, dl, loss):
    model.eval()

    loss_list = []
    infer_time = []
    psnr_list = []
    ssim_list = []
    rmse_list = []

    with torch.no_grad():
        for i, batch in enumerate(dl):
            input, target = batch['img'].to(device), batch['label'].to(device)

            start_time = time.time()
            output = model(input)
            infer_time.append(time.time() - start_time)

            batch_loss = loss(output, target)
            loss_list.append(batch_loss.item())

            batch_psnr_list, batch_ssim_list, batch_rmse_list = post_processing(target, output)
            psnr_list.extend(batch_psnr_list)
            ssim_list.extend(batch_ssim_list)
            rmse_list.extend(batch_rmse_list)

    mean_loss = np.mean(loss_list)

    mean_psnr, std_psnr = np.mean(psnr_list), np.std(psnr_list)
    mean_ssim, std_ssim = np.mean(ssim_list), np.std(ssim_list)
    mean_rmse, std_rmse = np.mean(rmse_list), np.std(rmse_list)

    mean_infer_time = np.mean(infer_time)

    return mean_loss, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_rmse, std_rmse, mean_infer_time