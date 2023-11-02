from typing import Tuple
import torch
import torchmetrics
import pytorch_lightning as pl
from cnn2d import CNN2D, ResNet2D
from cnn3d import CNN3D, ResNet3D
import numpy as np 
from dataset import load_coordinates
import os
import matplotlib.pyplot as plt
import cv2 


class LandmarkDetector(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.lr = cfg.lr
        self.cnn_type = cfg.cnn_type
        
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
                self.model = CNN3D(cfg.backbone_type, 10, cfg.output_dim, filters=30, kernel_size=4)
            elif cfg.cnn_type == 'resnet18':
                self.model = ResNet3D(cfg.backbone_type, 10, cfg.output_dim)
            print(f'[*] 3D data using {cfg.cnn_type} model with {cfg.backbone_type} backbone')
            
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
    
    def test_step(self, batch, batch_idx):
        mean_x, mean_y, x_coor, y_coor = self._test_preparation()
        img, target = batch
        size_image = img.shape[2]
        half_size = int(size_image/2)
        
        # predict
        y_hat = self.forward(img)
        y_hat = y_hat.detach().cpu().numpy()
        
        x_pred = int(y_hat[0][0]*size_image) + mean_x[0] - half_size
        y_pred = int(y_hat[0][1]*size_image) + mean_y[0] - half_size
    
        # calculate metrics
        distance = np.math.sqrt(((x_pred - x_coor) ** 2) + ((y_pred - y_coor) ** 2))
        horizontaldistance = np.math.sqrt(((x_pred - x_coor) ** 2))
        verticaldistance = np.math.sqrt(((y_pred - y_coor) ** 2))
        
        self.log('test.MRE', distance, on_epoch=True, on_step=False)
        self.log('test.horizontaldistance', horizontaldistance, on_epoch=True, on_step=False)
        self.log('test.verticaldistance', verticaldistance, on_epoch=True, on_step=False)

        # # plot the pred point
        # oringinal_image = cv2.imread('data/2D/S_Point/Images_without_test/image5.png')
        # target = target.detach().cpu().numpy()
        # img = img.detach().cpu().numpy()
        
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img[0][0], cmap='gray')
        # plt.scatter(y_hat[0][0]*size_image, y_hat[0][1]*size_image, c='r', alpha=1) 
        # plt.scatter(target[0][0]*size_image, target[0][1]*size_image, c='g', alpha=1) 
        # plt.title('roi view')
        
        # plt.subplot(1, 2, 2)
        # plt.imshow(oringinal_image, cmap='gray')
        # plt.scatter(x_pred, y_pred, c='r', alpha=1) 
        # plt.scatter(x_coor, y_coor, c='g', alpha=1)
        # plt.title('global view')
        
        # plt.suptitle('predited point in red, original point in green')
        # save_dir = os.path.join('predictions', self.cnn_type)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # plt.savefig(save_dir + '/image5.png')
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def _test_preparation(self):
        name_point = 'S_Point'
        text_mean_x_y_dir = 'data/Preparation/' + name_point + '/Mean_X_Y.txt'
        f = open(text_mean_x_y_dir, 'r')
        num_line = 0
        x_start_text = 'Mean_X:'
        x_end_text = ','
        y_start_text = 'Mean_Y:'
        y_end_text = ';'
        mean_x = []
        mean_y = []     
        for line in f:
            num_line += 1
            x_coor = line[line.find(x_start_text) + len(x_start_text): line.find(x_end_text)]
            y_coor = line[line.find(y_start_text) + len(y_start_text): line.find(y_end_text)]
            mean_x.append(int(x_coor))
            mean_y.append(int(y_coor))

        files = os.listdir('data/2D/S_Point/Coordinates_test/')
        for file in files:
            all_points = load_coordinates('data/2D/S_Point/Coordinates_test/' + file)
            x_coor, y_coor = all_points[0][0], all_points[0][1]
        return mean_x, mean_y, x_coor, y_coor