import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LandmarkDataset
from lightning_module import LandmarkDetector
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transform import RandomRotation, ColorJitter, GaussianBlur

import hydra
from omegaconf import DictConfig, OmegaConf

import warnings
warnings.filterwarnings("ignore")

def build_callbacks(cfg):
    
    dirpath = os.path.join(*['checkpoints', cfg.n_dim, cfg.cnn_type])
    checkpoint_callback = ModelCheckpoint(
        monitor='val.loss',
        mode='min',
        dirpath=dirpath,
        filename='{epoch:02d}',
        save_top_k=1,
        save_last=False,
        )
    
    early_stop_callback = EarlyStopping(
        monitor='val.loss',
        mode='min',
        patience=10,
    )
    
    callbacks = [checkpoint_callback, 
                 early_stop_callback,
                 ]
    return callbacks

def build_transforms():
    t = transforms.Compose([
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
        GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])
    
    return t

# @hydra.main(config_path="config", config_name="2d")
@hydra.main(config_path="config", config_name="3d")
def main(cfg: DictConfig):
    max_epochs = cfg.max_epochs
    batch_size = cfg.batch_size
    transforms = build_transforms()
    
    exp_name = f'{cfg.n_dim}_{cfg.cnn_type}'
    backbone_type = cfg.get('backbone_type', None)
    if backbone_type is not None:
        exp_name += f'_{backbone_type}backbone'
    exp_name += f'_lr{str(cfg.lr)}'
    
    logger = WandbLogger(project='landmark_detection', name=exp_name)
    # logger = None
    
    model = LandmarkDetector(cfg)    
    
    trainset = LandmarkDataset(cfg.train_path, cfg.img_dir, cfg.n_dim, cfg.point_type, cfg.size_image, cfg.mean_x, cfg.mean_y, transform=None)
    valset = LandmarkDataset(cfg.val_path, cfg.img_dir, cfg.n_dim, cfg.point_type, cfg.size_image, cfg.mean_x, cfg.mean_y, transform=None)
    testset = LandmarkDataset(cfg.test_path, cfg.img_dir, cfg.n_dim, cfg.point_type, cfg.size_image, cfg.mean_x, cfg.mean_y, transform=None)
    print(f'Length of trainset: {len(trainset)}')
    print(f'Length of valset: {len(valset)}')
    print(f'Length of testset: {len(testset)}')
    
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False
    )
    val_dataloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    test_dataloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    
    accelerator = 'gpu'
    # accelerator = 'mps'
    # if cfg.n_dim == '3d' and cfg.backbone_type == '3d':
    #     accelerator = 'cpu'
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        log_every_n_steps=1,
        precision=32,
        callbacks=build_callbacks(cfg),
        logger=logger,
        )    

    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_dataloader
                )
    # ckpt = 'checkpoints/3d/custom/epoch=19.ckpt'
    trainer.test(model, 
                 dataloaders=test_dataloader, 
                #  ckpt_path=ckpt,
                 )
    

if __name__ == '__main__':
    main()