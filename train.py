import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LandmarkDataset
from lightning_module import LandmarkDetector


def train():
    max_epochs = 400
    batch_size = 1
    
    model = LandmarkDetector(cnn_type='custom')    
    
    train_loader = DataLoader(
        LandmarkDataset(name_point='S_Point', size_image=520, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    trainer = pl.Trainer(
        accelerator='mps',
        max_epochs=max_epochs,
        log_every_n_steps=1,
        precision=32,
        )    

    trainer.fit(model, train_loader)
    

if __name__ == '__main__':
    train()