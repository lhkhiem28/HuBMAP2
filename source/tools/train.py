
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from imports import *
warnings.filterwarnings("ignore")

lightning.seed_everything(22)
from data import CellDataset
from engines import train_fn

df = pandas.read_csv("../dataset/meta.csv")
image_paths, mask_paths = sorted(glob.glob("../../dataset/npys/images_512/*.npy")), sorted(glob.glob("../../dataset/npys/masks_512/*.npy"))
train_image_paths, train_mask_paths = [path for path in image_paths if int(path.split("/")[-1].split("_")[0]) in df[df["is_training"] == True].id.values.tolist()], [path for path in mask_paths if int(path.split("/")[-1].split("_")[0]) in df[df["is_training"] == True].id.values.tolist()]
val_image_paths, val_mask_paths = [path for path in image_paths if int(path.split("/")[-1].split("_")[0]) in df[df["is_training"] == False].id.values.tolist()], [path for path in mask_paths if int(path.split("/")[-1].split("_")[0]) in df[df["is_training"] == False].id.values.tolist()]

loaders = {
    "train": torch.utils.data.DataLoader(
        CellDataset(
            train_image_paths, train_mask_paths
            , is_training = True
        ), 
        num_workers = 8, batch_size = 64
        , shuffle = True
    ), 
    "val": torch.utils.data.DataLoader(
        CellDataset(
            val_image_paths, val_mask_paths
            , is_training = False
        ), 
        num_workers = 8, batch_size = 64
        , shuffle = False
    ), 
}

model = seg.Unet(
    encoder_name = "resnext50_32x4d", encoder_weights = "ssl", 
    in_channels = 3, classes = 1
)
criterion, optimizer = seg.losses.DiceLoss("binary"), optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 5e-5)

wandb.login()
wandb.init(
    project = "HuBMAP2", name = "vanilla", 
    config = {
        "num_gpus": 3, 
        "num_epochs": 3, 
    }
)
save_ckp_path = "../ckps/{}".format("vanilla")
if not os.path.exists(save_ckp_path):
    os.makedirs(save_ckp_path)
train_fn(
    loaders, model, 
    criterion, optimizer
    , scheduler = None
    , save_ckp_path = save_ckp_path, training_verbose = True, 
)
wandb.finish()