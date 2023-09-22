from lightning import pytorch as pl
from utils.dataset import CTSegmentationDataset
from core.models import UNet
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def train(model, dataset, batch_size=4):
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    print(f'{train_len=}, {val_len=}')
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    trainer = pl.Trainer(max_epochs=2, log_every_n_steps=1, callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', patience=10)
    ])
    trainer.fit(model, train_dataloader, val_dataloader)


def main():
    model = UNet(n_channels=1, n_classes=3)
    dataset = CTSegmentationDataset('./data/segmentation_data')
    train(model, dataset)


if __name__ == '__main__':
    main()

