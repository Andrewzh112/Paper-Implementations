from torch import nn


class ResBlock(nn.Module):
    def __init__(self, input_channels, activation=None, norm=None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        if norm is None:
            norm = nn.InstanceNorm2d(input_channels)

        conv1 = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )
        conv2 = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )
        conv_block = [
            conv1,
            norm,
            activation,
            conv2,
            norm
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ContractingBlock(nn.Module):
    def __init__(self, input_channels, normalize=True, kernel_size=3, activation='leakyrelu'):
        super(ContractingBlock, self).__init__()
        self.conv = nn.Conv2d(
            input_channels,
            input_channels * 2,
            kernel_size=kernel_size,
            padding=1,
            stride=2,
            padding_mode='reflect'
        )
        self.activation = nn.LeakyReLU(0.2) if activation == 'leakyrelu' else nn.ReLU()
        if normalize:
            self.norm = nn.InstanceNorm2d(input_channels * 2)
        self.normalize = normalize

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = self.activation(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, normalize=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            input_channels,
            input_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        if normalize:
            self.norm = nn.InstanceNorm2d(input_channels // 2)
        self.normalize = normalize
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = self.activation(x)
        return x


class Feature(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        return self.conv(x)


def initialize_weights(model, nonlinearity='leaky_relu'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity=nonlinearity
            )
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)
