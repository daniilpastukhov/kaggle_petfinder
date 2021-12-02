from typing import Tuple
from pathlib import Path
import logging

import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from skimage.io import imread
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from cnn import CNN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PetDataset(Dataset):
    def __init__(self, image_paths, df, transforms) -> None:
        self.path_names = image_paths
        self.targets = None
        if 'Pawpularity' in df.columns:
            self.targets = df[df.index.isin([p.stem for p in self.path_names])]['Pawpularity'].values
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.path_names)

    def __getitem__(self, index: int) -> Tuple:
        image = imread(self.path_names[index])
        image = self.transforms(image=image)['image']
        if self.targets is None:
            return image

        return image, self.targets[index]


def train():
    LOAD_PATH = Path('data')

    logger = WandbLogger('petfinder_cnn', project='petfinder')

    train_df = pd.read_csv(Path(LOAD_PATH, 'train.csv'), index_col='Id')
    test_df = pd.read_csv(Path(LOAD_PATH, 'test.csv'), index_col='Id')

    train_image_paths = list(Path(LOAD_PATH, 'train').glob('*'))
    test_image_paths = list(Path(LOAD_PATH, 'test').glob('*'))

    img_size = 64

    conv_filters = [3, 8, 16, 32, 64]
    conv_kernels = [5, 5, 5, 5]
    conv_strides = [2, 2, 2, 2]

    train_transforms = A.Compose([
        A.Resize(img_size, img_size)
    ])

    val_transforms = A.Compose([
        A.Resize(img_size, img_size)
    ])

    train_dataset = PetDataset(train_image_paths, train_df, train_transforms)
    test_dataset = PetDataset(test_image_paths, test_df, val_transforms)

    val_size = int(len(train_dataset) * 0.8)
    train_set, val_set = torch.utils.data.random_split(train_dataset, lengths=[len(train_dataset) - val_size, val_size])

    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNN(img_size, conv_filters, conv_kernels, conv_strides)

    callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        filename=CNN.__name__
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=150,
        logger=logger,
        callbacks=[callback]
    )

    trainer.fit(model, train_loader, val_loader)
    prediction = trainer.predict(model, test_loader)
    torch.save(prediction, 'sub.pt')
    logger.finalize('ok')


if __name__ == '__main__':
    train()
