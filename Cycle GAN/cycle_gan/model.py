from torch import nn
from .ops import ResBlock, Feature, ContractingBlock, ExpandingBlock, initialize_weights


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim=64, n_res_blocks=9, weights_loc=None):
        super().__init__()
        feature = Feature(input_channels, hidden_dim)
        contract1 = ContractingBlock(hidden_dim, activation='relu')
        contract2 = ContractingBlock(hidden_dim * 2, activation='relu')
        gen_model = [feature, contract1, contract2]
        for _ in range(n_res_blocks):
            gen_model.append(ResBlock(hidden_dim * 4))
        expand1 = ExpandingBlock(hidden_dim * 4)
        expand2 = ExpandingBlock(hidden_dim * 2)
        downfeature = Feature(hidden_dim, output_channels)
        activation = nn.Tanh()
        gen_model += [expand1, expand2, downfeature, activation]
        self.gen = nn.Sequential(*gen_model)
        if weights_loc is None:
            initialize_weights(self, nonlinearity='relu')
        else:
            self.load_state_dict(weights_loc)

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_dim, weights_loc=None):
        super().__init__()
        upfeature = Feature(input_channels, hidden_dim)
        contract1 = ContractingBlock(hidden_dim, normalize=False, kernel_size=4)
        contract2 = ContractingBlock(hidden_dim * 2, kernel_size=4)
        contract3 = ContractingBlock(hidden_dim * 4, kernel_size=4)
        conv = nn.Conv2d(hidden_dim * 8, 1, kernel_size=1)
        disc_model = [upfeature, contract1, contract2, contract3, conv]
        self.disc = nn.Sequential(*disc_model)
        if weights_loc is None:
            initialize_weights(self)
        else:
            self.load_state_dict(weights_loc)

    def forward(self, x):
        return self.disc(x)
