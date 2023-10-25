import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LandmarkDataset
from lightning_module import LandmarkDetector
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import hydra
from omegaconf import DictConfig, OmegaConf


def build_callbacks(cfg):
    
    dirpath = os.path.join(*['checkpoints', cfg.dimension, cfg.cnn_type])
    checkpoint_callback = ModelCheckpoint(
        monitor='val.loss',
        mode='min',
        dirpath=dirpath,
        filename='{epoch:02d}',
        save_top_k=1,
        save_last=True,
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
        transforms.ToTensor(),
    ])
    
    return t

@hydra.main(config_path="config", config_name="2d")
def train(cfg: DictConfig):
    max_epochs = cfg.max_epochs
    batch_size = cfg.batch_size
    transforms = build_transforms()
    
    logger = WandbLogger(project='landmark_detection', name='2D_'+ cfg.cnn_type)
    # logger = None
    
    model = LandmarkDetector(cnn_type=cfg.cnn_type, output_dim=2)    
    
    train_loader = DataLoader(
        LandmarkDataset(name_point='S_Point', stage='train', size_image=520, transform=transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_dataloader = DataLoader(
        LandmarkDataset(name_point='S_Point', stage='val', size_image=520, transform=transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    trainer = pl.Trainer(
        accelerator='mps',
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
    

if __name__ == '__main__':
    train()