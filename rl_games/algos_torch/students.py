from rl_games.algos_torch.network_builder import ConvBlock, ResidualBlock
import torch
from torch import Tensor
import torch.nn as nn


class Impala2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, image_size: int,
                 state_size: int, action_size: int, activation: str = 'elu',
                 use_bn: bool = False, use_zero_init: bool = False) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, use_bn)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels,
                                        activation=activation,
                                        use_bn=use_bn,
                                        use_zero_init=use_zero_init)
        self.res_block2 = ResidualBlock(out_channels,
                                        activation=activation,
                                        use_bn=use_bn,
                                        use_zero_init=use_zero_init)

        image_mlp_in = int(out_channels * ((image_size / 2) ** 2))
        print("image_mlp_in:", image_mlp_in)

        self.image_mlp = nn.Sequential(
            nn.Linear(image_mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )


    def image_forward(self, image) -> Tensor:
        emb = self.conv(image)
        emb = self.max_pool(emb)
        emb = self.res_block1(emb)
        emb = self.res_block2(emb)
        emb = self.image_mlp(emb.flatten(1, -1))
        return emb

    def state_forward(self, state) -> Tensor:
        emb = self.state_mlp(state)
        return emb

    def forward(self, image, state):
        print('image.shape:', image.shape)
        image_emb = self.image_forward(image)
        print("image_embedding.shape:", image_emb.shape)
        state_emb = self.state_forward(state)

        emb = torch.cat([image_emb, state_emb], dim=-1)

        action = self.head(emb)
        return action


