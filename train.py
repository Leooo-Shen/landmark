import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LandmarkDataset
from lightning_module import LandmarkDetector
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

def build_callbacks():
    checkpoint_callback = ModelCheckpoint(
        monitor='val.loss',
        mode='min',
        dirpath='checkpoints',
        filename='{epoch:02d}-{val.acc_top5:.2f}',
        save_top_k=1,
        save_last=True,
        )
    
    # early_stop_callback = EarlyStopping(
    #     monitor='val.loss',
    #     mode='min',
    #     patience=5,
    # )
    
    callbacks = [checkpoint_callback, 
                #  early_stop_callback,
                 ]
    return callbacks

def build_transforms():
    t = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return t


def train():
    max_epochs = 50
    batch_size = 1
    logger = WandbLogger(project='landmark_detection', name='2D landmark')
    # logger = None
    transforms = build_transforms()
    
    model = LandmarkDetector(cnn_type='custom')    
    
    train_loader = DataLoader(
        LandmarkDataset(name_point='S_Point', stage='train', size_image=520, transform=transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    # val_dataloader = DataLoader(
    #     LandmarkDataset(name_point='S_Point', stage='val', size_image=520, transform=transforms),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=False
    # )
    
    trainer = pl.Trainer(
        accelerator='mps',
        max_epochs=max_epochs,
        log_every_n_steps=1,
        precision=32,
        callbacks=build_callbacks(),
        logger=logger,
        )    

    trainer.fit(model, 
                train_dataloaders=train_loader, 
                # val_dataloaders=val_dataloader
                )
    

if __name__ == '__main__':
    train()