import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=5, stride=1):
    """
    Determine the output size after applying convolution operation.
    """
    return (size - (kernel_size - 1) - 1) // stride + 1


class CNN(pl.LightningModule):
    def __init__(self, img_size, conv_filters, conv_kernels, conv_strides):
        super().__init__()

        self.fc_input_size_h = img_size
        self.fc_input_size_w = img_size

        # Conv layers
        self.conv_layers = []
        for i in range(len(conv_filters) - 1):
            self.conv_layers.append(nn.Conv2d(in_channels=conv_filters[i],
                                              out_channels=conv_filters[i + 1],
                                              kernel_size=conv_kernels[i],
                                              stride=conv_strides[i]))
            self.conv_layers.append(nn.BatchNorm2d(conv_filters[i + 1]))
            self.conv_layers.append(nn.ReLU())
            self.fc_input_size_h = conv2d_size_out(self.fc_input_size_h, conv_kernels[i], 2)
            self.fc_input_size_w = conv2d_size_out(self.fc_input_size_w, conv_kernels[i], 2)

        self.conv_layers = nn.Sequential(*self.conv_layers)

        # Fully connected layers
        fc_layers = [self.fc_input_size_h * self.fc_input_size_w * conv_filters[-1], 1]
        self.fc_layers = []

        for i in range(len(fc_layers) - 1):
            self.fc_layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))

        self.fc_layers = nn.Sequential(nn.Flatten(), *self.fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def common_step(self, batch, batch_id):
        images, targets = batch
        images = images.to(self.device).permute(0, 3, 1, 2).float()
        target = targets.to(self.device).float()
        output = self(images)
        loss = F.mse_loss(torch.flatten(output), target)
        return loss

    def training_step(self, batch, batch_id):
        loss = self.common_step(batch, batch_id)
        self.log('train/loss', loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_id):
        loss = self.common_step(batch, batch_id)
        self.log('val/loss', loss)
        return loss
