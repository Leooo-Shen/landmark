import torchvision
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, output_dim=None, filters=None, kernel_size=None):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.backbone = nn.Sequential(
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
            
    def forward(self, x):
        x = self.input_layer(x)
        return self.backbone(x)



class ResNet2D(nn.Module):
    def __init__(self, output_dim, pretrained=False):
        super().__init__()
        
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        resnet.fc = nn.Linear(512, output_dim)
        
        # conver 1 channel to 3 channels
        self.input_layer = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        self.backbone = resnet
            
    def forward(self, x):
        x = self.input_layer(x)
        return self.backbone(x)


# test
if __name__ == '__main__':
    from torchsummary import summary
    
    # model = CNN2D(output_dim=2, filters=30, kernel_size=4)
    model = ResNet2D(output_dim=2)
    # summary(model, (1, 520, 520))
    print(model.backbone.conv1)