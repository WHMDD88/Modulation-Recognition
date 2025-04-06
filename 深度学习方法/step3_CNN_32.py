import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle

def get_data():
    with open('../output_data2/abs_cyclic_spectrum32.pickle','rb') as f:
        data_dict=pickle.load(f)
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

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    epoch_losses=[]
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 计算该 epoch 的平均损失
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')
    return epoch_losses


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
    #cm_diaply = ConfusionMatrixDisplay(cm)
    #cm_diaply.plot()
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

    metrics_df = pd.DataFrame(
        columns=['SNR', 'train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score'])
    all_losses_df = pd.DataFrame()  # 用于存储所有信噪比下的 epoch 损失

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
        model = CNNClassifier(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # 训练模型
        num_epochs = 50
        epoch_losses=train_model(model, train_loader, criterion, optimizer, num_epochs)
        # 保存模型
        #torch.save(model.state_dict(), 'modulation_classifier.pth')

        #测试模型
        train_acc, _,_, _, _, _l=test_model(model,train_loader)
        print(f'train_acc: {train_acc}%')
        test_acc, cm,auc1, f1, precision, recall=test_model(model,test_loader)
        print(f'test_acc: {test_acc}%')
        print(f'AUC: {auc1}')
        print(f'F1 Score: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')

        print('**************************************************************************')
        print()

        # 将当前信噪比下的指标添加到 DataFrame 中
        metrics_df = metrics_df.append({
            'SNR': dB,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'roc_auc_score': auc1,
            'f1_score': f1,
            'precision_score': precision,
            'recall_score': recall
        }, ignore_index=True)

        # 将当前信噪比下的 epoch 损失添加到 all_losses_df 中
        losses_df = pd.DataFrame({f'SNR_{dB}': epoch_losses})
        all_losses_df = pd.concat([all_losses_df, losses_df], axis=1)

    metrics_df.to_csv('output_data4\\epoch50_CNN32.csv', index=False)
    all_losses_df.to_csv('output_data4\\epoch50_CNN_losses32.csv', index=False)