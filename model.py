import torchvision
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, cnn_type, output_dim=None, filters=None, kernel_size=None):
        super().__init__()
        
        if cnn_type == 'custom':
            self.model = nn.Sequential(
                nn.Conv2d(1, filters, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(filters, filters * 2, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(filters * 2, filters * 3, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(filters * 3, filters * 4, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(filters * 4, filters * 5, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(filters * 5, filters * 6, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(filters * 6, filters * 7, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(filters * 7),
                nn.MaxPool2d(2),
                nn.Conv2d(filters * 7, filters * 8, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(filters * 8, filters * 8, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(filters * 8, filters * 16, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(filters * 16, filters * 16, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(filters * 16, 600),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(600, 400),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(400, 500),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(500, 400),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(300, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, output_dim),
            )
            
        elif cnn_type == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=False)
            resnet.fc = nn.Identity()
            self.model = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0),
                resnet,
                nn.Linear(512, output_dim),
            )
            
    def forward(self, x):
        return self.model(x)
    
# test
if __name__ == '__main__':
    from torchsummary import summary
    
    # model = CNN('custom', output_dim=2, filters=30, kernel_size=4)
    model = CNN('resnet18', output_dim=2)
    summary(model, (1, 520, 520))