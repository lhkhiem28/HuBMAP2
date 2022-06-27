
import os, sys
from imports import *

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, 
        image_paths, mask_paths
        , augment = None
    ):
        self.image_paths, self.mask_paths = image_paths, mask_paths
        self.augment = augment
        self.transform = A.Compose([
            A.Resize(512, 512, p = 1), 
            AT.ToTensorV2(), 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image, mask = np.load(self.image_paths[index]), np.load(self.mask_paths[index])
        if self.augment is not None:
            augmented = self.augment(image = image, mask = mask)
            image, mask = augmented["image"], augmented["mask"]

        image = standardize(image)
        transformed = self.transform(image = image, mask = mask)
        image, mask = transformed["image"], transformed["mask"]
        return image, mask