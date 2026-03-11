import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def lr_finder(device, model, loss_fn, train_dl):
    init_lr = 1e-5
    final_lr = 1
    num_steps = len(train_dl)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    lrs = []
    losses = []

    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    lr = init_lr

    model.train()

    for batch in tqdm(train_dl):
        # Set LR dynamically
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad()
        inputs, targets = batch['img'].to(device), batch['label'].to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())

        # Stop if loss explodes
        if loss.item() > 4 * losses[0]:
            break

        lr *= lr_mult
        if lr > final_lr:
            break

    df = pd.DataFrame({'lrs': lrs, 'losses': losses})
    df.to_csv('lr_range_finder.csv', index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Smoothed Loss")
    plt.title("LR Finder Curve")
    plt.grid(True)
    plt.show()

