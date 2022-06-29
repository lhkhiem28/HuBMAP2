
import os, sys
from imports import *

def train_fn(
    loaders, model, 
    criterion, optimizer
    , scheduler = None
    , save_ckp_path = None, training_verbose = True, 
):
    print("\nStart Training ...\n" + " = "*16)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids = list(range(wandb.config.num_gpus)))

    best_dice = 0.0
    for epoch in tqdm.tqdm(range(1, wandb.config.num_epochs + 1), disable = training_verbose):
        if training_verbose:
            print("epoch {:2}/{:2}".format(epoch, wandb.config.num_epochs) + "\n" + "-"*16)

        model.train()
        running_loss, running_dice = 0.0, 0.0
        for images, masks in tqdm.tqdm(loaders["train"]):
            images, masks = images.cuda().float(), masks.cuda().float()

            logits = model(images)
            loss, dice = criterion(logits, masks), 1 - seg.losses.DiceLoss("binary")(logits, masks)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss, running_dice = running_loss + loss.item()*images.size(0), running_dice + dice.item()*images.size(0)

        if scheduler is not None:
            scheduler.step()

        epoch_loss, epoch_dice = running_loss/len(loaders["train"].dataset), running_dice/len(loaders["train"].dataset)
        wandb.log({"loss": epoch_loss, "dice": epoch_dice})
        if training_verbose:
            print("{:<5} | ".format("train") + "loss: {:.4f}, dice: {:.4f}".format(epoch_loss, epoch_dice))