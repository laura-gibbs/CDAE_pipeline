import torch.nn as nn
import torch


class CDAE(nn.Module):
    def __init__(self, num_blocks=4, input_ch=1, base_ch=8, dropout=False):
        super().__init__()
        if base_ch is None:
            base_ch = 8
        self.encoder0 = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch),
            nn.ReLU(),
        )
        self.encoder1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=base_ch, out_channels=base_ch*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch*2, out_channels=base_ch*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*2),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=base_ch*2, out_channels=base_ch*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch*4, out_channels=base_ch*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*4),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=base_ch*4, out_channels=base_ch*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch*8, out_channels=base_ch*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*8),
            nn.ReLU(),
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=base_ch*8, out_channels=base_ch*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch*8, out_channels=base_ch*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*8),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=base_ch*16, out_channels=base_ch*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch*8, out_channels=base_ch*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*4),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=base_ch*8, out_channels=base_ch*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch*4, out_channels=base_ch*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*2),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=base_ch*4, out_channels=base_ch*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch*2, out_channels=base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch),
            nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(in_channels=base_ch*2, out_channels=base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_ch),
            nn.ReLU(),
        )
        if dropout:
            self.fully_connected = nn.Sequential(
                nn.Conv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=1),
                nn.Dropout(),
                nn.Conv2d(in_channels=base_ch, out_channels=1, kernel_size=1),
            )
        else:
            self.fully_connected = nn.Sequential(
                nn.Conv2d(in_channels=base_ch, out_channels=1, kernel_size=1),
            )

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec1 = self.decoder1(torch.cat((self.up(enc4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.up(dec1), enc2), dim=1))
        dec3 = self.decoder3(torch.cat((self.up(dec2), enc1), dim=1))
        dec4 = self.decoder4(torch.cat((self.up(dec3), enc0), dim=1))

        out = self.fully_connected(dec4)
        return out





