from typing import Tuple
import torch
import torchmetrics
import pytorch_lightning as pl
from model import CNN

class LandmarkDetector(pl.LightningModule):
    def __init__(self, cnn_type, output_dim=2, lr=1e-4) -> None:
        super().__init__()
        self.lr = lr
        if cnn_type == 'custom':
            self.model = CNN(cnn_type, output_dim, filters=30, kernel_size=4)
        elif cnn_type == 'resnet18':
            self.model = CNN(cnn_type, output_dim)
                        
        self.criterion = torch.nn.MSELoss()
        self.mae_train = torchmetrics.MeanAbsoluteError()
        self.mae_val = torchmetrics.MeanAbsoluteError()
            
        
    def forward(self, x):
        return self.model(x)
    
    
    def _shared_step(self, batch, batch_idx, stage=None) -> torch.Tensor:
        """
        Shared step for train and val
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        y_hat = y_hat.detach()

        if stage == 'train':
            self.mae_train(y_hat, y)
            self.log('train.loss', loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log('train.mae', self.mae_train, on_epoch=True, on_step=False, prog_bar=True)
        
        elif stage == 'val':
            self.mae_val(y_hat, y)
            self.log('val.loss', loss, on_epoch=True, on_step=False)
            self.log('val.mae', self.mae_val, on_epoch=True, on_step=False)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer