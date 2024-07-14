from lightning import pytorch as pl
from utils.dataset import CTSegmentationDataset
from core.models import UNet
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


def train(model, dataset, n_epochs=100, batch_size=4, train_fraction=0.8):
    train_len = int(train_fraction * len(dataset))
    val_len = len(dataset) - train_len

    print(f'\nTrain dataset size = {train_len}, Validation dataset size = {val_len}\n')
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  persistent_workers=True, num_workers=2)

    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=10),
                 ModelCheckpoint(save_top_k=2, monitor='val_loss')]

    trainer = pl.Trainer(max_epochs=n_epochs, callbacks=callbacks, log_every_n_steps=20)

    trainer.fit(model, train_dataloader, val_dataloader)

    return train_dataset, val_dataset


def main():
    model = UNet(n_channels=1, n_classes=3)
    dataset = CTSegmentationDataset('./data/segmentation_data')
    train(model, dataset, batch_size=4)


if __name__ == '__main__':
    main()
