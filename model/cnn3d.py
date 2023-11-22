import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cnn2d import CNN2D, ResNet2D
from model.resnet_helper import generate_model

class CNN3D(nn.Module):
    def __init__(self, backbone_type, in_channels, output_dim, filters=None, kernel_size=None):
        super().__init__()
        assert backbone_type in ['2d', '3d']
        self.backbone_type = backbone_type
        # treat 3D input as channels, use 2D conv
        if backbone_type == '2d':
            self.backbone = CNN2D(output_dim, filters, kernel_size)
            # to modify the channel
            self.backbone.input_layer = nn.Identity()
            self.input_layer = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        # use 3D convolution, 1 channel, 10 slices
        elif backbone_type == '3d':
            self.input_layer = nn.Sequential(
                nn.Conv3d(1, filters, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool3d((1,2,2))
            )
            self.backbone = nn.Sequential(
                nn.Conv3d(filters, filters * 2, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool3d((1,2,2)),
                nn.Conv3d(filters * 2, filters * 3, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.Conv3d(filters * 3, filters * 4, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool3d((1,2,2)),
                nn.Conv3d(filters * 4, filters * 5, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.Conv3d(filters * 5, filters * 6, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm3d(filters * 6),
                nn.MaxPool3d((1,2,2)),
                nn.Conv3d(filters * 6, filters * 7, kernel_size=(2, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.Conv3d(filters * 7, filters * 8, kernel_size=(2, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool3d((2,2,2)),
                nn.Conv3d(filters * 8, filters * 16, kernel_size=(2, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.Conv3d(filters * 16, filters * 16, kernel_size=(2, kernel_size, kernel_size), stride=1, padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),
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
        if self.backbone_type == '2d':
            x = x.squeeze(1)
        x = self.input_layer(x)
        return self.backbone(x)
    

class ResNet3D(nn.Module):
    def __init__(self, backbone_type, in_channels, output_dim):
        super().__init__()
        assert backbone_type in ['2d', '3d']
        self.backbone_type = backbone_type
        
        # treat 3D input as channels, use 2D conv
        if backbone_type == '2d':
            self.backbone = ResNet2D(output_dim)
            self.backbone.input_layer = nn.Identity()
            self.backbone.backbone.conv1 = nn.Identity()
            self.input_layer = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        elif backbone_type == '3d':
            self.input_layer = nn.Conv3d(1, 64, kernel_size=(7,7,7), stride=(1,2,2), padding=(3,3,3), bias=False)
            self.backbone = generate_model(model_depth=18)
            self.backbone.conv1 = nn.Identity()
            self.backbone.fc =  nn.Linear(512, output_dim)
           
    def forward(self, x):
        if self.backbone_type == '2d':
            x = x.squeeze(1)
        x = self.input_layer(x)
        return self.backbone(x)
    

# test
if __name__ == '__main__':
    from torchsummary import summary
    import torch
    x = torch.randn([2, 1, 10, 520, 520]) # [C, T, W, H]
    
    # model = CNN3D(backbone_type='3d', in_channels=10, output_dim=2, filters=30, kernel_size=4)
    model = ResNet3D(backbone_type='3d', in_channels=10, output_dim=2)
    
    summary(model, (1, 10, 520, 520))
    # print(model(x).shape)