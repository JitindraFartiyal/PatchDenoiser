import numpy as np
import torch
import matplotlib.pyplot as plt
import pydicom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn

from ReDen.datasets import get_dataloaders
from model import ReDen
from pre_post_processing import post_processing
from utilities import load_ct_normalized
import tqdm


def sort_by_metric(outputs, targets, psnr_list, ssim_list, loss_list, global_fused_list, metric='psnr'):
    combined = list(zip(outputs, targets, psnr_list, ssim_list, loss_list, global_fused_list))
    if metric == 'psnr':
        combined_sorted = sorted(combined, key=lambda x: x[2])  # x[2] = PSNR value
    else:
        combined_sorted = sorted(combined, key=lambda x: x[3])

    outputs_sorted, targets_sorted, psnr_sorted, ssim_sorted, loss_sorted, sorted_global_fused = zip(*combined_sorted)

    return list(outputs_sorted), list(targets_sorted), list(psnr_sorted), list(ssim_sorted), list(loss_sorted), list(sorted_global_fused)



def evaluate(device, model, test_loader, is_global_fused=True):
    model.eval()
    outputs = []
    targets = []
    psnr_list = []
    ssim_list = []
    loss_list = []
    global_fused_list = []
    criterion = nn.L1Loss()

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_loader)):
            input, target = batch['img'].to(device, non_blocking=True), batch['label'].to(device, non_blocking=True)
            if is_global_fused:
                output, global_fused = model(input)
                global_fused = global_fused.mean(dim=1).squeeze(0).detach().cpu().numpy()
                global_fused_list.append(global_fused)
            else:
                output = model(input)
            loss = criterion(output, target)
            loss_list.append(loss.item())

            psnr, ssim, _ = post_processing(target, output)
            outputs.append(output.squeeze(0).squeeze(0).detach().cpu().numpy())
            targets.append(target.squeeze(0).squeeze(0).detach().cpu().numpy())
            psnr_list.append(psnr[0].item())
            ssim_list.append(ssim[0].item())

        mean_psnr = np.mean(list(psnr_list))
        std_psnr = np.std(list(psnr_list))
        mean_ssim = np.mean(list(ssim_list))
        std_ssim = np.std(list(ssim_list))

        print('Mean PSNR: {:.4f}, SSIM: {:.4f}'.format(mean_psnr, mean_ssim))
        print('Std PSNR: {:.4f} SSIM: {:.4f}'.format(std_psnr, std_ssim))

    if is_global_fused:
        return outputs, targets, psnr_list, ssim_list, loss_list, global_fused_list
    else:
        return outputs, targets, psnr_list, ssim_list, loss_list

if __name__ == '__main__':
    train_labels_filepath = r"D:\data\Project_Denoising\ldct\Training_Image_Data\train.csv"
    test_labels_filepath = r"D:\data\Project_Denoising\ldct\Training_Image_Data\test.csv"

    val_dl = get_dataloaders(train_labels_filepath, test_labels_filepath, test_batch_size=1, only_test=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReDen().to(device)
    model.load_state_dict(torch.load(r'saved_models/model.pth'))

    outputs, targets, psnr_list, ssim_list, loss_list, global_fused_list = evaluate(device, model, val_dl)

    sorted_outputs, sorted_targets, sorted_psnr, sorted_ssim, sorted_loss, sorted_gb = sort_by_metric(outputs, targets, psnr_list, ssim_list, loss_list, global_fused_list)

    i = 0
    print(sorted_psnr[i], sorted_ssim[i], sorted_loss[i], sorted_outputs[i].shape, sorted_targets[i].shape)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(sorted_targets[i], cmap='gray')
    ax[1].imshow(sorted_outputs[i], cmap='gray')
    # ax[2].imshow(sorted_targets[i] - sorted_outputs[i])
    ax[2].imshow(sorted_gb[i], cmap='gray')
    plt.show()

