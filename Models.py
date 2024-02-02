import torch
import torch.nn as nn


class LinearGenerator(nn.Module):
    def __init__(self, noise_dim, image_dim):
        super(LinearGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)



class ConvolutionalGenerator(nn.Module):
    def __init__(self, noise_dim, image_channels):
        super(ConvolutionalGenerator, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(noise_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)  # Flatten the image
        return self.main(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)  # No activation
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.fc(x)

