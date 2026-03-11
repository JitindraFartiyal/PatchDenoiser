import gc
import multiprocessing

import pandas as pd
import time
import torch

from datasets import get_dataloaders
from loss import loss_fn1
from model import PatchUp
from train_and_test import train_one_epoch, validate

if __name__ == "__main__":
    train_labels_filepath = r"D:\data\Project_Denoising\ldct\Training_Image_Data\3mm B30\labels\fold_1_train.csv"
    test_labels_filepath = r"D:\data\Project_Denoising\ldct\Training_Image_Data\3mm B30\labels\fold_1_test.csv"

    device = torch.device('cuda')

    batch_sizes = [1, 2, 4, 8, 16]
    num_workers = [1, 2, 4, 8, 16]

    loss_fn = loss_fn1()

    for batch_size in batch_sizes:
        for num_worker in num_workers:
            if num_worker > batch_size:
                continue
            model = PatchUp().to(device)
            model.eval()
            optim = torch.optim.Adam(model.parameters(), lr=4e-5)
            train_dl, val_dl = get_dataloaders(train_labels_filepath, test_labels_filepath,
                                               batch_size=batch_size,
                                               num_workers=num_worker,
                                               test_batch_size=batch_size,
                                               test_num_workers=num_worker)

            it = iter(val_dl)
            _ = next(it)  # cold batch (ignored)
            warm_batch = next(it)  # warm batch to warm GPU transfer
            warm_images = warm_batch['img'].to(device)
            warm_target = warm_batch['label'].to(device)

            with torch.no_grad():
                _ = model(warm_images)

            start_epoch = time.time()
            epoch_loss = train_one_epoch(
                epoch=0,
                epochs=1,
                device=device,
                model=model,
                dl=train_dl,
                optim=optim,
                loss=loss_fn,
                prints=False
            )
            torch.cuda.synchronize()
            epoch_time = time.time() - start_epoch

            print(f"BS={batch_size}, Workers={num_worker}, Epoch time: {epoch_time:.4f}s")
            _, _, _, _, _, _, _, mean_infer_time = validate(device, model, val_dl, loss_fn)

            del model, train_dl, val_dl, warm_batch, warm_images, warm_target
            gc.collect()
            torch.cuda.empty_cache()
