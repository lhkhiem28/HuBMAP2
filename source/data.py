
import os, sys
from imports import *

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, 
        image_paths, mask_paths
        , is_training
    ):
        self.image_paths, self.mask_paths = image_paths, mask_paths
        self.is_training = is_training

        self.augment = A.Compose([
            A.HorizontalFlip(), A.VerticalFlip(), 
        ])
        self.transform = A.Compose([
            A.Resize(
                512, 512, 
            ), 
            AT.ToTensorV2(), 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image, mask = np.load(self.image_paths[index]), np.load(self.mask_paths[index])
        if self.is_training:
            augmented = self.augment(image = image, mask = mask)
            image, mask = augmented["image"], augmented["mask"]

        image = standardize(image)
        transformed = self.transform(image = image, mask = mask)
        image, mask = transformed["image"], transformed["mask"]
        return image, mask