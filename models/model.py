import torch
import torch.nn as nn
import torch.nn.functional as F


class DAVE2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (150, 200)

        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)

        self.dropout = nn.Dropout()

        self.lin1 = nn.Linear(in_features=13824, out_features=100, bias=True)
        self.lin2 = nn.Linear(in_features=100, out_features=50, bias=True)
        self.lin3 = nn.Linear(in_features=50, out_features=10, bias=True)
        # self.lin4 = nn.Linear(in_features=10, out_features=2, bias=True)
        self.lin4 = nn.Linear(in_features=10, out_features=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x, inplace=True)

        x = self.conv2(x)
        x = F.elu(x, inplace=True)

        x = self.conv3(x)
        x = F.elu(x, inplace=True)

        x = self.conv4(x)
        x = F.elu(x, inplace=True)

        x = self.conv5(x)
        x = F.elu(x, inplace=True)

        x = x.flatten(1)

        x = self.lin1(x)
        x = self.dropout(x)
        x = F.elu(x, inplace=True)

        x = self.lin2(x)
        x = self.dropout(x)
        x = F.elu(x, inplace=True)

        x = self.lin3(x)
        x = self.dropout(x)
        x = F.elu(x, inplace=True)

        x = self.lin4(x)
        x = torch.tanh(x)

        return x
