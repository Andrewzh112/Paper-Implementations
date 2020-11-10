import torch
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F


class AlexNet(nn.Module):
    def __init__(self, flatten_dim=9216, fc_hidden=4096, out_dim=1000, dropout=0.5):
        super(AlexNet, self).__init__()
        
        self.flatten_dim = flatten_dim
        self.p = dropout

        self.conv1 = nn.Conv2d(
                        in_channels=3,
                        out_channels=96,
                        kernel_size=11,
                        stride=4,
                        padding=2
                    )
        self.pool1 = nn.MaxPool2d(
                        kernel_size=3,
                        stride=2
                    )
        self.rm1 = nn.LocalResponseNorm(
                        size=5,
                        alpha=1e-4,
                        beta=0.75,
                        k=2
                    )
        
        self.conv2 = nn.Conv2d(
                        in_channels=96,
                        out_channels=256,
                        kernel_size=5,
                        padding=2
                    )
        self.pool2 = nn.MaxPool2d(
                        kernel_size=3,
                        stride=2
                    )
        self.rm2 = nn.LocalResponseNorm(
                        size=5,
                        alpha=1e-4,
                        beta=0.75,
                        k=2
                    )
        
        self.conv3 = nn.Conv2d(
                        in_channels=256,
                        out_channels=384,
                        kernel_size=3,
                        padding=1
                    )
        
        self.conv4 = nn.Conv2d(
                        in_channels=384,
                        out_channels=384,
                        kernel_size=3,
                        padding=1
                    )

        self.conv5 = nn.Conv2d(
                        in_channels=384,
                        out_channels=256,
                        kernel_size=3,
                        padding=1
                    )
        self.pool5 = nn.MaxPool2d(
                        kernel_size=3,
                        stride=2
                    )
        
        self.fc1 = nn.Linear(flatten_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc3 = nn.Linear(fc_hidden, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """
        'We initialized the weights in each layer from a zero-mean Gaussian 
        distribution with standard de-viation 0.01.  We initialized the neuron 
        biases in the second, fourth, and fifth convolutional layers,as well as 
        in the fully-connected hidden layers, with the constant 1. We initialized 
        the neuronbiases in the remaining layers with the constant 0.'
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.conv3.weight.data.normal_(0, 0.01)
        self.conv5.weight.data.normal_(0, 0.01)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2.weight.data.normal_(0 0.01)
        self.fc3.weight.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1)
        self.conv4.bias.data.fill_(1)
        self.conv5.bias.data.fill_(1)
    
    @amp.autocast()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.rm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.rm2(x)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        x = x.view(-1, self.flatten_dim)

        x = F.dropout(F.relu(self.fc1(x)), self.p)
        x = F.dropout(F.relu(self.fc2(x)), self.p)
        output = self.fc3(x)
        return output