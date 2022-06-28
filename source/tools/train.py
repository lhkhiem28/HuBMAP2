
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from imports import *
warnings.filterwarnings("ignore")

lightning.seed_everything(22)
from data import CellDataset

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
        num_workers = 8, batch_size = 224
        , shuffle = True
    ), 
    "val": torch.utils.data.DataLoader(
        CellDataset(
            val_image_paths, val_mask_paths
            , is_training = False
        ), 
        num_workers = 8, batch_size = 224
        , shuffle = False
    ), 
}

model = seg.Unet(
    encoder_name = "resnext50_32x4d", encoder_weights = "ssl", 
    in_channels = 3, classes = 1
)