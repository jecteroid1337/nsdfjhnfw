import lightning.pytorch as pl
from .models_parts import *
from .losses import dice_loss


class UNet(pl.LightningModule):
    def __init__(self, n_channels=1, n_classes=3, *, bilinear=True, dice_loss_impact=0.0):
        super().__init__()

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.dice_loss_impact = dice_loss_impact

        scaling_factor = 2 if bilinear else 1

        self.initial = DoubleConv(self.n_channels, 64)
        self.down1 = UNetDownBlock(64, 128)
        self.down2 = UNetDownBlock(128, 256)
        self.down3 = UNetDownBlock(256, 512)
        self.down4 = UNetDownBlock(512, 1024 // scaling_factor)

        self.up1 = UNetUpBlock(512 + 512, 512 // scaling_factor, bilinear)
        self.up2 = UNetUpBlock(256 + 256, 128, bilinear)
        self.up3 = UNetUpBlock(128 + 128, 64, bilinear)
        self.up4 = UNetUpBlock(64 + 64, 64, bilinear)
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss_function = F.cross_entropy if self.n_classes > 1 else F.binary_cross_entropy
        loss = loss_function(y_hat, y)

        self.training_step_outputs.append((y_hat, loss))

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([loss for (_, loss) in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()
        print(f'| Train_loss: {avg_loss:.2f}')
        self.log('train_loss', avg_loss, logger=False, prog_bar=True)


    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)

        self.validation_step_outputs.append((y_hat, loss))

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([loss for (_, loss) in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        print(f'[Epoch {self.trainer.current_epoch + 1:3}] Val_loss: {avg_loss:.2f}', end=' ')
        self.log('val_loss', avg_loss, logger=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='min',
                                                                  factor=0.2,
                                                                  patience=5,
                                                                  verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }

        return [optimizer], [lr_dict]
