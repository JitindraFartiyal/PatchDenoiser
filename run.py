import torch
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataloaders
from model import PatchUp
from train_and_test import train_one_epoch, validate
from loss import loss_fn1

from utilities import get_model_stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(23)

torch.backends.cudnn.benchmark = True

def evaluate(model, dataloader, loss_fn, device, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    val_start_time = time.time()
    val_loss, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_rmse, std_rmse, infer_time = validate(device, model,
                                                                                                   dataloader, loss_fn)

    print(
        "Val loss: {:.5f}, psnr: {:.3f}/{:.3f}, ssim: {:.3f}/{:.3f}, rmse: {:.3f}/{:.3f},"
        "val time: {:.5f} best psnr/ssim [{:.3f}/{:.3f} {:.3f}/{:.3f}]".format(
            val_loss, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_rmse, std_rmse, infer_time,
            time.time() - val_start_time,
            mean_psnr, std_psnr, mean_ssim, std_ssim))

    dummy_input = torch.randn(1, 1, 512, 512).to(device)
    for _ in range(10):
        _ = model(dummy_input)

    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(100):
        start = time.time()
        _ = model(dummy_input)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print("Average time: {:.4f}".format(avg_time))



def start_training():
    train_labels_filepath = r"D:\data\Project_Denoising\ldct\Training_Image_Data\3mm B30\labels\fold_5_train.csv"
    test_labels_filepath = r"D:\data\Project_Denoising\ldct\Training_Image_Data\3mm B30\labels\fold_5_test.csv"

    # train_labels_filepath = r"D:\data\Project_Denoising\ldct\Gen_Test_Image_Data\processed_data\train.csv"
    # test_labels_filepath = r"D:\data\Project_Denoising\ldct\Gen_Test_Image_Data\processed_data\test.csv"

    epochs = 40
    val_every = 1
    model_name = r"test_model.pth"
    writer = SummaryWriter(log_dir="./runs/test_run/")

    train_dl, val_dl = get_dataloaders(train_labels_filepath, test_labels_filepath,
                                       batch_size=1, num_workers=1,
                                       test_batch_size=1, test_num_workers=1)
    print("Train dl: ", len(train_dl), "Val dl: ", len(val_dl))
    model = PatchUp().to(device)

    dummy_input = torch.randn(1, 1, 512, 512).to(device)
    get_model_stats(model, dummy_input=dummy_input)


    # Weight Initialization
    for layer in model.modules():
        if isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    loss_fn = loss_fn1()
    # print(f"Starting lr range finder...")
    # lr_finder(device, model, loss_fn, train_dl)
    # exit(0)
    best_lr = 1e-2
    eta_min = best_lr/160
    print(f"Chosen best lr: {best_lr}")
    optim = torch.optim.Adam(model.parameters(), lr=best_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=epochs,
        eta_min=eta_min,
    )

    # evaluate(model, val_dl, loss_fn, device, "saved_models/" + model_name)
    # exit(0)

    best_epoch = 0
    best_mean_psnr = -float('inf')
    best_std_psnr = -float('inf')
    best_mean_ssim = -float('inf')
    best_std_ssim = -float('inf')
    best_mean_rmse = -float('inf')
    best_std_rmse = -float('inf')
    epoch_time = []

    print("Starting training")
    for epoch in range(epochs):

        start_time = time.time()
        epoch_loss = train_one_epoch(epoch, epochs, device, model, train_dl, optim, loss_fn, prints=True)
        epoch_time.append(time.time() - start_time)
        print(" [Train loss: {:.5f}]  [epoch time: {:.5f}]".format(epoch_loss, time.time() - start_time))
        scheduler.step()

        if epoch >= 29 and ((epoch+1) % val_every == 0 or epoch == 0 or epoch == epochs-1):
            val_start_time = time.time()
            val_loss, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_rmse, std_rmse, infer_time = validate(device, model, val_dl, loss_fn)


            writer.add_scalars('Loss/Epoch',
                               {'train_loss':epoch_loss,
                                'val_loss':val_loss}, epoch)
            writer.add_scalars('Psnr/Epoch', {'val_psnr':mean_psnr}, epoch)

            if mean_psnr > best_mean_psnr:
                best_mean_psnr = mean_psnr
                best_mean_ssim = mean_ssim
                best_mean_rmse = mean_rmse

                best_std_psnr = std_psnr
                best_std_ssim = std_ssim
                best_std_rmse = std_rmse

                best_epoch = epoch
                torch.save(model.state_dict(), r"saved_models/" + model_name)

            print(
                "Val loss: {:.5f}, psnr: {:.3f}/{:.3f}, ssim: {:.3f}/{:.3f}, rmse: {:.3f}/{:.3f}, infer_time: {:.5f}, "
                "val time: {:.5f} best psnr/ssim [{:.3f}/{:.3f} {:.3f}/{:.3f}]".format(
                    val_loss, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_rmse, std_rmse, infer_time, time.time() - val_start_time,
                    best_mean_psnr, best_std_psnr, best_mean_ssim, best_std_ssim))

    print("Best psnr: {:.3f}/{:.3f}, best ssim: {:.3f}/{:.3f}, best rmse: {:.3f}/{:.3f} at epoch: {}".format(
        best_mean_psnr, best_std_psnr, best_mean_ssim, best_std_ssim, best_mean_rmse, best_std_rmse, best_epoch))

    writer.flush()

if __name__ == '__main__':
    start_training()
    
