import torch
import time
import pandas as pd

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from pre_post_processing import pre_processing


class DenoisedDataset(Dataset):
    def __init__(self, labels_file, transform=None):
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, item):
        input_image = pre_processing(self.labels_df.iloc[item]['quarter_dose'])
        gt_image = pre_processing(self.labels_df.iloc[item]['full_dose'])

        # input_image = pre_processing(self.labels_df.iloc[item]['input_image_path'])
        # gt_image = pre_processing(self.labels_df.iloc[item]['gt_image_path'])

        if self.transform is not None:
            gt_image = self.transform(gt_image)
            input_image = self.transform(input_image)

        return {
            "img": input_image,
            "label": gt_image
        }


def get_dataloaders(train_labels_file, test_labels_file, batch_size=8, num_workers=8, test_batch_size=8, test_num_workers=8, only_test=False):
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32)
    ])

    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32)
    ])


    val_ds = DenoisedDataset(labels_file=test_labels_file,
                             transform=val_transforms)
    val_dl = DataLoader(val_ds,
                        batch_size=test_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=True
                        )
    if only_test:
        return val_dl


    train_ds = DenoisedDataset(labels_file=train_labels_file,
                               transform=train_transforms)
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=test_num_workers,
                          pin_memory=True,
                          drop_last=False,
                          persistent_workers=True
                          )


    return train_dl, val_dl

if __name__ == "__main__":
    train_labels_filepath = r"D:\data\Project_Denoising\ldct\Training_Image_Data\3mm B30\labels\fold_1_train.csv"
    test_labels_filepath = r"D:\data\Project_Denoising\ldct\Training_Image_Data\3mm B30\labels\fold_1_test.csv"

    start_time = time.time()
    train_dl, val_dl = get_dataloaders(train_labels_filepath, test_labels_filepath,
                                       batch_size=4,
                                       num_workers=1,
                                       test_batch_size=1,
                                       test_num_workers=1)
    print(f"Train dataset length: {len(train_dl)}, val dataset length: {len(val_dl)}, time: {time.time() - start_time}")

    iter_start_time = time.time()
    first_batch = next(iter(train_dl))
    print(f"Iter speed: {time.time() - iter_start_time}")

    iter_start_time = time.time()
    second_batch = next(iter(train_dl))
    print(f"Iter speed: {time.time() - iter_start_time}")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # img, label = t['img'].to(device), t['label'].to(device)
    # img = torch.squeeze(img, 1)[0].detach().cpu().numpy()
    # label = torch.squeeze(label, 1)[0].detach().cpu().numpy()
    # print(img.shape, label.shape)
    #
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(img, cmap='gray')
    # axs[1].imshow(label, cmap='gray')
    # plt.show()