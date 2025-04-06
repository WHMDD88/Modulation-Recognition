
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# 数据生成函数
def get_data():
    # 定义调制方式列表
    modulation_list = ['AM-DSB', 'AM-SSB', 'WBFM', 'BPSK', 'QPSK', '8PSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64']
    # 定义信噪比列表
    snr_list = [i for i in range(-20,19,2)]
    # 初始化一个空字典
    data_dict = {}

    # 遍历调制方式和信噪比的组合
    for modulation in modulation_list:
        for snr in snr_list:
            # 生成包含 1000 个形状为 (21, 32) 的 numpy 数组的列表
            array_list = [np.random.rand(21, 32) for _ in range(1000)]
            # 以调制方式和信噪比组成的元组作为键
            key = (modulation, snr)
            # 将列表作为值添加到字典中
            data_dict[key] = array_list
    return data_dict

# 获取指定信噪比的所有数据
def get_data_by_snr(data_dict, target_snr):
    result = {}
    for key, value in data_dict.items():
        _, snr = key
        if snr == target_snr:
            modulation = key[0]
            result[modulation] = value
    return result

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data = []
        self.labels = []
        label_mapping = {}
        label_index = 0
        for key, arrays in data_dict.items():
            modulation = key
            if modulation not in label_mapping:
                label_mapping[modulation] = label_index
                label_index += 1
            for array in arrays:
                self.data.append(array)
                self.labels.append(label_mapping[modulation])

        # 数据标准化
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

"""带注意力及机制的CNN"""
class AttentionCNN(nn.Module):
    def __init__(self, num_classes):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_input_dim = self._calculate_fc_input_dim()
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.relu3 = nn.ReLU()
        self.attention = nn.Linear(128, 1)
        self.softmax_att = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.fc_input_dim)
        x = self.relu3(self.fc1(x))
        att_weights = self.softmax_att(self.attention(x))
        x = att_weights * x
        # 避免求和操作，保持形状为 (batch_size, 128)
        x = x
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def _calculate_fc_input_dim(self):
        x = torch.randn(1, 1, 21, 32)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x.view(1, -1).size(1)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# 测试函数
def test_model(model, test_loader):
    model.eval()
    all_predicted = []
    all_labels = []
    all_outputs = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_predicted)
    cm_diaply = ConfusionMatrixDisplay(cm)
    cm_diaply.plot()
    # 计算 AUC 值（仅适用于二分类问题，多分类需要特殊处理）
    auc1 = roc_auc_score(all_labels, np.array(all_outputs),multi_class='ovo')
    # 计算 F1 分数、Precision 值和 Recall 值
    f1 = f1_score(all_labels, all_predicted, average='micro')
    precision = precision_score(all_labels, all_predicted, average='micro')
    recall = recall_score(all_labels, all_predicted, average='micro')
    return accuracy, cm,auc1,f1, precision, recall

# 主函数
if __name__ == "__main__":
    # 生成数据
    data_dict = get_data()

    dB_list=[i for i in range(-20,19,2)]
    for dB in dB_list:
        print("信噪比:",dB)
        data_dict1=get_data_by_snr(data_dict,dB)
        # 创建数据集和数据加载器
        dataset = CustomDataset(data_dict1)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # 初始化模型、损失函数和优化器
        num_classes = 11
        model = AttentionCNN(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # 训练模型
        num_epochs = 10
        train_model(model, train_loader, criterion, optimizer, num_epochs)
        # 保存模型
        #torch.save(model.state_dict(), 'modulation_classifier.pth')

        #测试
        train_acc, _,_, _, _, _l=test_model(model,train_loader)
        print(f'train_acc: {train_acc}%')
        #测试集
        test_acc, cm,auc1, f1, precision, recall=test_model(model,test_loader)
        print(f'test_acc: {test_acc}%')
        # 计算 AUC 值（仅适用于二分类问题，多分类需要特殊处理）
        print(f'AUC: {auc1}')
        # 计算 F1 分数、Precision 值和 Recall 值
        print(f'F1 Score: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print('**************************************************************************')
        print()
