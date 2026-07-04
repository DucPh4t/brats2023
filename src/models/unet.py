import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, init_features=32):
        super().__init__()
        self.inc = DoubleConv(n_channels, init_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))
        
        self.up1 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 8, init_features * 4)
        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        self.up3 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 2, init_features)
        
        self.outc = nn.Conv2d(init_features, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        return self.outc(x)  # [PAPER FIX] Trả về raw logits. Sigmoid được apply ở ngoài (trainer/evaluator).
