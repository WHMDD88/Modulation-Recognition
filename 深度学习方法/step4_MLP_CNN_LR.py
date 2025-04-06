import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# MLP 相关代码
class MultiClassNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate):
        super(MultiClassNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


def get_net(num_inputs, units_hiddens, num_outputs, dropout_rate):
    net = MultiClassNetwork(num_inputs, units_hiddens, num_outputs, dropout_rate)
    return net


def train(net, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss(reduction='none')
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            optimizer.step()
    return net


def preprocess_data(df, dB):
    df1 = df[df['SNR'] == dB]
    data_df = df1[['index', 'Modulation', 'SNR', 'SSSE', 'SSIE', 'PSSE', 'PEIE', 'max_pow_den', 'std']].copy()
    #只要 熵部分的特征
    X_data = data_df[['SSSE', 'SSIE', 'PSSE', 'PEIE', 'max_pow_den', 'std']].copy()
    mapping_dict = {
        'AM-DSB': 0,
        'AM-SSB': 1,
        'WBFM': 2,
        'BPSK': 3,
        'QPSK': 4,
        '8PSK': 5,
        'CPFSK': 6,
        'GFSK': 7,
        'PAM4': 8,
        'QAM16': 9,
        'QAM64': 10,
    }
    data_df.loc[:, 'Modulation_label'] = data_df['Modulation'].map(mapping_dict)
    Y_df = data_df['Modulation_label']
    # 保存索引，用于后续匹配
    indices = np.arange(len(X_data))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_data, Y_df, indices, test_size=0.3, random_state=42)
    train_features = torch.tensor(X_train.values, dtype=torch.float32)
    test_features = torch.tensor(X_test.values, dtype=torch.float32)
    train_labels = torch.tensor(y_train.values, dtype=torch.long)
    test_labels = torch.tensor(y_test.values, dtype=torch.long)
    return train_features, test_features, train_labels, test_labels, idx_train, idx_test


# CNN 相关代码
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
        self.data = np.array(self.data)
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
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_input_dim = self._calculate_fc_input_dim()
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(-1, self.fc_input_dim)
        x = self.fc(x)
        return x

    def _calculate_fc_input_dim(self):
        x = torch.randn(1, 1, 21, 32)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x.view(1, -1).size(1)


def train_cnn(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


# 堆叠法集成
def stacking_ensemble():
    try:
        df = pd.read_csv('../output_data1/nor_all_feas.csv', encoding='UTF-8')
    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
        return

    data_dict = get_data()
    dB_list = [i for i in range(-20, 19, 2)]

    metrics_df = pd.DataFrame(
        columns=['SNR', 'train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score'])
    all_losses_df = pd.DataFrame()  # 用于存储所有信噪比下的 epoch 损失
    for dB in dB_list:
        print("信噪比:", dB)

        # MLP 数据处理
        train_features_mlp, test_features_mlp, train_labels_mlp, test_labels_mlp, idx_train, idx_test = preprocess_data(df, dB)
        num_inputs = train_features_mlp.shape[1]
        num_outputs = 11
        units_hidden = 64
        num_epochs_mlp = 50
        lr = 0.01
        batch_size = 256
        weight_decay = 0
        net_mlp = get_net(num_inputs, units_hidden, num_outputs, 0.5)
        net_mlp = train(net_mlp, train_features_mlp, train_labels_mlp, num_epochs_mlp, lr, weight_decay, batch_size)

        # 获取 MLP 预测结果
        net_mlp.eval()
        with torch.no_grad():
            mlp_train_probs = net_mlp(train_features_mlp).numpy()
            mlp_test_probs = net_mlp(test_features_mlp).numpy()

        # CNN 数据处理
        data_dict1 = get_data_by_snr(data_dict, dB)
        dataset = CustomDataset(data_dict1)

        # 使用相同的索引划分训练集和测试集
        train_dataset = torch.utils.data.Subset(dataset, idx_train)
        test_dataset = torch.utils.data.Subset(dataset, idx_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        num_classes = 11
        model_cnn = CNNClassifier(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_cnn.parameters(), lr=0.01)
        num_epochs_cnn = 50
        model_cnn = train_cnn(model_cnn, train_loader, criterion, optimizer, num_epochs_cnn)

        # 获取 CNN 预测结果
        model_cnn.eval()
        cnn_train_probs = []
        cnn_test_probs = []
        with torch.no_grad():
            for inputs, _ in train_loader:
                outputs = model_cnn(inputs)
                cnn_train_probs.extend(outputs.numpy())
            for inputs, _ in test_loader:
                outputs = model_cnn(inputs)
                cnn_test_probs.extend(outputs.numpy())
        cnn_train_probs = np.array(cnn_train_probs)
        cnn_test_probs = np.array(cnn_test_probs)

        # 合并预测结果作为新特征
        train_new_features = np.hstack((mlp_train_probs, cnn_train_probs))
        test_new_features = np.hstack((mlp_test_probs, cnn_test_probs))

        # 训练元模型 逻辑回归
        meta_model = LogisticRegression()
        meta_model.fit(train_new_features, train_labels_mlp.numpy())

        # 元模型预测
        meta_hat=meta_model.predict(train_new_features)
        meta_hat_proba=meta_model.predict_proba(train_new_features)

        meta_predictions = meta_model.predict(test_new_features)
        meta_proba = meta_model.predict_proba(test_new_features)
        # 计算评估指标
        train_acc= accuracy_score(train_labels_mlp.numpy(), meta_hat)
        test_acc = accuracy_score(test_labels_mlp.numpy(), meta_predictions)
        auc1 = roc_auc_score(test_labels_mlp.numpy(), meta_proba, multi_class='ovo')
        f1 = f1_score(test_labels_mlp.numpy(), meta_predictions, average='micro')
        precision = precision_score(test_labels_mlp.numpy(), meta_predictions, average='micro')
        recall = recall_score(test_labels_mlp.numpy(), meta_predictions, average='micro')

        print(f"训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        print(f"AUC: {auc1:.4f}")
        print(f"F1 分数: {f1:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print('***********************************************************')
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

    metrics_df.to_csv('../output_data4/epoch100_MLPCNN.csv', index=False)

if __name__ == "__main__":
    stacking_ensemble()