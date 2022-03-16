import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 11, 4),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, 2),
                                   nn.BatchNorm2d(96))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, 2),
                                   nn.BatchNorm2d(256))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1, 2),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 384, 3, 1, 2),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, 2))
        self.conv6 = nn.Sequential(nn.Linear(256, 4096),
                                   nn.Dropout(0.5),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Linear(4096, 4096),
                                   nn.Dropout(0.5),
                                   nn.ReLU())
        self.out = nn.Linear(4096, 1000)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.out(x)
        return x

def test():
    net = AlexNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())