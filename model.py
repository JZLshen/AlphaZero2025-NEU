import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_ResidualBlock(nn.Module):
    def __init__(self, num_filters, reduction=16):
        super(SE_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SELayer(num_filters, reduction)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class GomokuNet(nn.Module):
    def __init__(self, game):
        super(GomokuNet, self).__init__()
        self.board_size = game.get_board_size()[0]
        self.action_size = game.get_action_size()

        # 输入通道为3: 水平边，垂直边，格子归属
        self.conv_in = nn.Conv2d(
            3, CONV_FILTERS, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(CONV_FILTERS)
        self.relu_in = nn.ReLU(inplace=True)

        self.res_blocks = nn.ModuleList(
            [SE_ResidualBlock(CONV_FILTERS) for _ in range(RESIDUAL_BLOCKS)]
        )

        self.policy_conv = nn.Conv2d(CONV_FILTERS, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        # 策略头的输出大小需要与总动作数匹配
        self.policy_fc = nn.Linear(
            2 * self.board_size * self.board_size, self.action_size
        )

        self.value_conv = nn.Conv2d(CONV_FILTERS, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_size * self.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def to_input_tensor(self, board):
        """Converts board representation to a single tensor for the network."""
        h_lines, v_lines, box_owners = board

        # Pad all planes to the same size (n x n) for easy stacking
        h_padded = np.pad(h_lines, ((0, 0), (0, 1)), "constant")
        v_padded = np.pad(v_lines, ((0, 1), (0, 0)), "constant")
        box_padded = np.pad(box_owners, ((0, 1), (0, 1)), "constant")

        # Stack them into a single tensor
        # NOTE: Make sure the batching logic in main.py can handle this.
        # This function is for a single board.
        input_tensor = torch.tensor(
            np.array([h_padded, v_padded, box_padded]), dtype=torch.float32
        )
        return input_tensor.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        x = x.to(DEVICE)
        out = self.relu_in(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        pi = F.relu(self.policy_bn(self.policy_conv(out)))
        pi = pi.view(pi.size(0), -1)
        pi = self.policy_fc(pi)
        pi = F.log_softmax(pi, dim=1)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return pi, v
