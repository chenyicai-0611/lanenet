import torch
import torch.nn as nn
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#新建一个简单的leNet
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)
    def forward(self, input):
        output = self.conv1(input)
        output = nn.ReLU()(output)
        output = self.conv2(output)
        output = nn.ReLU()(output)
        output = self.pool(output)
        output = self.conv3(output)
        output = nn.ReLU()(output)
        output = self.conv4(output)
        output = nn.ReLU()(output)
        output = output.view(-1, 16 * 16 * 24)
        output = self.fc(output)
        return output
model = SimpleNet().to(device=device)

import torch.nn.utils.prune as prune
#  用prune剪枝
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.conv4, 'weight'),
    (model.fc, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)