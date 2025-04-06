import torch.nn as nn
import torch
from torchsummary import summary  # 新增导入
import netron

# CNN 分类器类
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        # 第一个卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二个卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 计算全连接层的输入维度
        self.fc_input_dim = self._calculate_fc_input_dim()
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 前向传播
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(-1, self.fc_input_dim)  # 展平
        x = self.fc(x)
        return x

    def _calculate_fc_input_dim(self):
        # 计算全连接层的输入维度
        x = torch.randn(1, 1, 21, 32)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x.view(1, -1).size(1)
if __name__ == '__main__':
    model=CNNClassifier(11)
    input=torch.ones((1,21,32))
    torch.onnx.export(model,input ,f='CNN.onnx')
    netron.start('CNN.onnx')