from typing import Tuple
import torch
import torchmetrics
import pytorch_lightning as pl
from model.cnn2d import CNN2D, ResNet2D
from model.cnn3d import CNN3D, ResNet3D
import numpy as np 
from dataset import load_coordinates
import os
import matplotlib.pyplot as plt
import cv2 
import wandb
from PIL import Image

class LandmarkDetector(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.lr = cfg.lr
        self.cnn_type = cfg.cnn_type
        self.save_dir = os.path.join(*['predictions', cfg.n_dim, self.cnn_type + f'_{cfg.backbone_type}backbone'])
        
        # 2D detector
        if cfg.n_dim == '2d':
            if cfg.cnn_type == 'custom':
                self.model = CNN2D(cfg.output_dim, filters=30, kernel_size=4)
            elif cfg.cnn_type == 'resnet18':
                self.model = ResNet2D(cfg.output_dim, pretrained=False)
            print(f'[*] 2D data using {cfg.cnn_type} model')
        
        # 3D detector
        elif cfg.n_dim == '3d':
            if cfg.cnn_type == 'custom':
                self.model = CNN3D(cfg.backbone_type, cfg.n_slice, cfg.output_dim, filters=30, kernel_size=4)
            elif cfg.cnn_type == 'resnet18':
                self.model = ResNet3D(cfg.backbone_type, cfg.n_slice, cfg.output_dim)
            print(f'[*] 3D data using {cfg.cnn_type} model with {cfg.backbone_type} backbone')
        
        self.n_dim = cfg.n_dim
        self.criterion = torch.nn.MSELoss()
        self.mae_train = torchmetrics.MeanAbsoluteError()
        self.mae_val = torchmetrics.MeanAbsoluteError()
        
        self.mse_test = torchmetrics.MeanSquaredError()
        self.mae_test_x = torchmetrics.MeanAbsoluteError()
        self.mae_test_y = torchmetrics.MeanAbsoluteError()
        self.test_step_outputs = []
        
        if cfg.n_dim == '3d':
            self.mae_test_z = torchmetrics.MeanAbsoluteError()
        
        self.mean_x = cfg.mean_x
        self.mean_y = cfg.mean_y
        
    def forward(self, x):
        return self.model(x)
    
    
    def _shared_step(self, batch, batch_idx, stage=None) -> torch.Tensor:
        """
        Shared step for train and val
        """
        img, target = batch
        pred = self.forward(img)
        loss = self.criterion(pred, target)
        pred = pred.detach()

        if stage == 'train':
            self.mae_train(pred, target)
            self.log('train.loss', loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log('train.mae', self.mae_train, on_epoch=True, on_step=False, prog_bar=True)
        
        elif stage == 'val':
            self.mae_val(pred, target)
            self.log('val.loss', loss, on_epoch=True, on_step=False)
            self.log('val.mae', self.mae_val, on_epoch=True, on_step=False)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        img, target = batch
        size_image = img.shape[-1]
        half_size = int(size_image / 2)
        
        pred = self.forward(img)
        
        if self.n_dim == '2d':
            pred = (pred * size_image).int()
            target = (target * size_image).int()
        elif self.n_dim == '3d':
            # rescale x
            pred[:, 0] = pred[:, 0] * size_image
            pred[:, 1] = pred[:, 1] * size_image
            # rescale y
            target[:, 0] = target[:, 0] * size_image
            target[:, 1] = target[:, 1] * size_image
            # rescale z
            pred[:, 2] = pred[:, 2] * 16
            target[:, 2] = target[:, 2] * 16
            pred = pred.int()
            target = target.int()
            
        self.test_step_outputs.append({'image': img.detach().cpu().numpy(), 
                                       'target': target.detach().cpu().numpy(), 
                                       'pred': pred.detach().cpu().numpy()
                                       })
    
    def on_test_epoch_end(self):
        preds = []
        targets = []
        for output in self.test_step_outputs:
            preds.append(output['pred'])
            targets.append(output['target'])
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        mre, std, distances = self.metrics(preds, targets)   
        
        z_distance = None
        if len(distances) == 2:
            x_distance, y_distance = distances
        elif len(distances) == 3:
            x_distance, y_distance, z_distance = distances
            
        self.log('test.MRE', mre, on_epoch=True, on_step=False)
        self.log('test.MRE_std', std, on_epoch=True, on_step=False)
        self.log('test.horizontal_distance', x_distance, on_epoch=True, on_step=False)
        self.log('test.vertical_distance', y_distance, on_epoch=True, on_step=False)
        if z_distance is not None:
            self.log('test.slice_distance', z_distance, on_epoch=True, on_step=False)
        
        # plot the first batch
        output = self.test_step_outputs[0]
        img = output['image']
        target = output['target']
        pred = output['pred']
        batch_size = img.shape[0]
        
        # visualize
        for i in range(batch_size):
            self.plot_results(img[i], target[i], pred[i], self.save_dir, idx=i)
        print(f'[*] Visualize the first batch of testset in {self.save_dir}')
        
    def plot_results(self, img, target, pred, save_dir, idx=None):
        if len(img.shape) == 3:
            img = img[0]
        elif len(img.shape) == 4:
            img = img[0][0]
        size_image = img.shape[0]
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.scatter(pred[0], pred[1], c='blue') 
        plt.scatter(target[0], target[1], c='r') 
        plt.title('predited point in blue, original point in red')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{str(idx)}.png'))
    
    def metrics(self, pred, target):
        delta = pred - target
        delta_x = delta[:, 0]
        delta_y = delta[:, 1]
        x_distance = np.abs(delta_x).mean().item()
        y_distance = np.abs(delta_y).mean().item()
        distances = [x_distance, y_distance]
        
        if self.n_dim == '2d':
            D = np.sqrt(delta_x ** 2 + delta_y ** 2)
        elif self.n_dim == '3d':
            delta_z = delta[:, 2]
            z_distance = np.abs(delta_z).mean().item()
            D = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
            distances.append(z_distance)
            
        mre = D.mean().item()
        std = D.std().item()
        
        return mre, std, distances
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    