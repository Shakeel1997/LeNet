import torch
import torch.nn as nn
from torchsummary import summary


class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x


# # Print model summary
# model = LeNet()
# summary(model.cuda(), (3, 28, 28)) # for 32*32 image padding = 0
# # print(model)


# Test
# def test_lenet():
#     x = torch.randn(16, 1, 28, 28)
#     # x = x.to(torch.float32)
#     model = LeNet()
#     return model(x)
#
#
# if __name__ == "__main__":
#     out = test_lenet()
#     print(out.shape)
