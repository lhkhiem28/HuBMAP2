
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
            loss, dice = criterion(logits, masks), 1 - seg.losses.DiceLoss("binary", from_logits = False)(torch.round(F.logsigmoid(logits.detach().cpu()).exp()), masks.detach().cpu())

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss, running_dice = running_loss + loss.item()*images.size(0), running_dice + dice.item()*images.size(0)

        if scheduler is not None:
            scheduler.step()

        epoch_loss, epoch_dice = running_loss/len(loaders["train"].dataset), running_dice/len(loaders["train"].dataset)
        wandb.log({"train_loss": epoch_loss, "train_dice": epoch_dice})
        if training_verbose:
            print("{:<5} | ".format("train") + "loss: {:.4f}, dice: {:.4f}".format(epoch_loss, epoch_dice))

        with torch.no_grad():
            model.eval()
            running_loss, running_dice = 0.0, 0.0
            for images, masks in tqdm.tqdm(loaders["val"]):
                images, masks = images.cuda().float(), masks.cuda().float()

                logits = model(images)
                loss, dice = criterion(logits, masks), 1 - seg.losses.DiceLoss("binary", from_logits = False)(torch.round(F.logsigmoid(logits.detach().cpu()).exp()), masks.detach().cpu())

                running_loss, running_dice = running_loss + loss.item()*images.size(0), running_dice + dice.item()*images.size(0)

        epoch_loss, epoch_dice = running_loss/len(loaders["val"].dataset), running_dice/len(loaders["val"].dataset)
        wandb.log({"val_loss": epoch_loss, "val_dice": epoch_dice})
        if training_verbose:
            print("{:<5} | ".format("val") + "loss: {:.4f}, dice: {:.4f}".format(epoch_loss, epoch_dice))

        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(model.module, "{}/best.ptl".format(save_ckp_path))

    print("\nValidation ...\n" + " = "*16)
    model = torch.load("{}/best.ptl".format(save_ckp_path), map_location = "cuda")
    model = nn.DataParallel(model, device_ids = list(range(wandb.config.num_gpus)))

    varied_dices, varied_thresholds = [], [round(_, 2) for _ in np.arange(1, 10)*0.1]
    for threshold in tqdm.tqdm(varied_thresholds):
        with torch.no_grad():
            model.eval()
            running_dice = 0.0
            for images, masks in loaders["val"]:
                images, masks = images.cuda().float(), masks.cuda().float()

                logits = model(images)
                dice = 1 - seg.losses.DiceLoss("binary", from_logits = False)(torch.where(F.logsigmoid(logits.detach().cpu()).exp() > threshold, 1.0, 0.0), masks.detach().cpu())

                running_dice = running_dice + dice.item()*images.size(0)

        varied_dices.append(running_dice/len(loaders["val"].dataset))
    optimal_dice, optimal_thres = max(varied_dices), varied_thresholds[varied_dices.index(max(varied_dices))]
    print("Optimal | " + "optimal_dice: {:.4f}, optimal_thres: {:.4f}".format(optimal_dice, optimal_thres))
    print("\nFinish !!!\n")