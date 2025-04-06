import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle


def ant_colony_optimization(mlp_probs, cnn_probs, true_labels, num_ants=20, num_iterations=50, alpha=1, beta=2, rho=0.5, Q=1):
    num_classes = mlp_probs.shape[1]
    num_params = 2 * num_classes
    # 初始化信息素矩阵
    pheromone = np.ones((num_params,))

    best_params = None
    best_loss = float('inf')
    aco_losses = []  # 记录蚁群算法每次迭代的损失

    for iteration in range(num_iterations):
        solutions = []
        losses = []

        # 每只蚂蚁构建一个解决方案
        for _ in range(num_ants):
            # 随机生成一个初始解
            params = np.random.rand(num_params)
            # 确保满足约束条件
            ai = params[:num_classes]
            bi = params[num_classes:]
            total = np.sum(ai + bi)
            params = params / total * num_classes

            loss = objective_function(params, mlp_probs, cnn_probs, true_labels)
            solutions.append(params)
            losses.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_params = params

        # 更新信息素
        pheromone = (1 - rho) * pheromone
        for i in range(num_ants):
            pheromone += Q / losses[i] * solutions[i]

        # 记录本次迭代的最小损失
        aco_losses.append(best_loss)
        #print(f"蚁群算法第 {iteration + 1} 次迭代，最小损失: {best_loss:.4f}")

    optimal_ai = best_params[:num_classes]
    optimal_bi = best_params[num_classes:]
    return optimal_ai, optimal_bi, aco_losses


def settings_random_seed():
    seed = 28
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

# MLP 模型相关部分
class MultiClassNetwork(nn.Module):
    """自定义block"""

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


def get_k_fold_data(k, i, X, y):
    """获取 训练集，验证集"""
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)

    return X_train, y_train, X_valid, y_valid


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss(reduction='none')
    mlp_losses = []  # 记录 MLP 每次迭代的损失

    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            optimizer.step()
            epoch_loss += l.mean().item()

        epoch_loss /= len(train_iter)
        mlp_losses.append(epoch_loss)
        #print(f"MLP 第 {epoch + 1} 次迭代，损失: {epoch_loss:.4f}")

    return net, mlp_losses

def k_fold(net, k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    """10折训练开始"""
    all_mlp_losses = []
    for i in range(k):
        # 得到每一折的训练集,验证集
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, X_train, y_train)
        # 训练num_epochs
        net, mlp_losses = train(net, X_train, y_train, X_valid, y_valid,
                                num_epochs, learning_rate, weight_decay,
                                batch_size)
        all_mlp_losses.extend(mlp_losses)

    return net, all_mlp_losses


def preprocess_data(df, dB):
    df1 = df[df['SNR'] == dB]
    data_df = df1[['index', 'Modulation', 'SNR', 'SSSE', 'SSIE', 'PSSE', 'PEIE', 'max_pow_den', 'std']].copy()
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
    # 生成索引
    indices = np.arange(len(X_data))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)

    train_features = torch.tensor(X_data.iloc[train_indices].values, dtype=torch.float32)
    test_features = torch.tensor(X_data.iloc[test_indices].values, dtype=torch.float32)
    train_labels = torch.tensor(Y_df.iloc[train_indices].values, dtype=torch.long)
    test_labels = torch.tensor(Y_df.iloc[test_indices].values, dtype=torch.long)
    return train_features, test_features, train_labels, test_labels, train_indices, test_indices



# CNN 模型相关部分
def get_data():
    with open('../output_data2/abs_cyclic_spectrum32.pickle', 'rb') as f:
        data_dict = pickle.load(f)
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
    cnn_losses = []  # 记录 CNN 每次迭代的损失

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        cnn_losses.append(epoch_loss)
        #print(f"CNN 第 {epoch + 1} 次迭代，损失: {epoch_loss:.4f}")

    return model, cnn_losses


# 联合模型部分
def objective_function(params, mlp_probs, cnn_probs, true_labels):
    num_classes = len(params) // 2
    ai = params[:num_classes]
    bi = params[num_classes:]
    combined_probs = np.maximum(ai * mlp_probs, bi * cnn_probs)
    combined_probs = combined_probs / combined_probs.sum(axis=1, keepdims=True)  # 归一化
    return log_loss(true_labels, combined_probs)

# 约束条件函数，这里在蚁群算法中直接在生成解时处理约束
def constraint(params):
    num_classes = len(params) // 2
    ai = params[:num_classes]
    bi = params[num_classes:]
    return np.sum(ai + bi) - num_classes


def calculate_metrics(mlp_probs, cnn_probs, optimal_ai, optimal_bi, true_labels):
    combined_probs = np.maximum(optimal_ai * mlp_probs, optimal_bi * cnn_probs)
    combined_probs = combined_probs / combined_probs.sum(axis=1, keepdims=True)  # 归一化
    predictions = np.argmax(combined_probs, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, combined_probs, multi_class='ovo')
    f1 = f1_score(true_labels, predictions, average='micro')
    precision = precision_score(true_labels, predictions, average='micro')
    recall = recall_score(true_labels, predictions, average='micro')
    return accuracy, auc, f1, precision, recall

if __name__ == "__main__":
    settings_random_seed()

    try:
        df = pd.read_csv('../output_data1/nor_all_feas.csv', encoding='UTF-8')
    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
        exit(1)
    data_dict = get_data()
    dB_list = [i for i in range(-20, 19, 2)]
    metrics_df = pd.DataFrame(
        columns=['SNR', 'train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score'])
    loss_df = pd.DataFrame()  # 用于存储每个信噪比下的损失

    for dB in dB_list:
        print("信噪比:", dB)

        # MLP 模型训练和预测
        train_features_mlp, test_features_mlp, train_labels_mlp, test_labels_mlp, train_indices, test_indices = preprocess_data(df, dB)
        num_inputs = train_features_mlp.shape[1]
        num_outputs = 11
        units_hidden = 64
        k, num_epochs_mlp, lr, batch_size, weight_decay = 5, 5, 0.01, 256, 0
        net_mlp = get_net(num_inputs, units_hidden, num_outputs, 0.5)
        net_mlp, mlp_losses = k_fold(net_mlp, k, train_features_mlp, train_labels_mlp,
                                     num_epochs_mlp, lr, weight_decay, batch_size)

        net_mlp.eval()
        with torch.no_grad():
            mlp_train_probs = net_mlp(train_features_mlp).numpy()
            mlp_test_probs = net_mlp(test_features_mlp).numpy()

        # CNN 模型训练和预测
        data_dict1 = get_data_by_snr(data_dict, dB)
        dataset = CustomDataset(data_dict1)

        # 使用相同的索引划分训练集和测试集
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        num_classes = 11
        model_cnn = CNNClassifier(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_cnn.parameters(), lr=0.001)
        num_epochs_cnn = 5
        model_cnn, cnn_losses = train_model(model_cnn, train_loader, criterion, optimizer, num_epochs_cnn)

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

        # 使用蚁群算法优化可信度因子
        optimal_ai, optimal_bi, aco_losses = ant_colony_optimization(mlp_train_probs, cnn_train_probs, train_labels_mlp.numpy())
        print(f"信噪比 {dB} 下的最优 ai: {optimal_ai}")
        print(f"信噪比 {dB} 下的最优 bi: {optimal_bi}")

        # 计算训练集指标
        train_acc, train_auc, train_f1, train_precision, train_recall = calculate_metrics(
            mlp_train_probs, cnn_train_probs, optimal_ai, optimal_bi, train_labels_mlp.numpy())

        # 计算测试集指标
        test_acc, test_auc, test_f1, test_precision, test_recall = calculate_metrics(
            mlp_test_probs, cnn_test_probs, optimal_ai, optimal_bi, test_labels_mlp.numpy())

        print(f"训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        print(f"训练集 AUC: {train_auc:.4f}")
        print(f"测试集 AUC: {test_auc:.4f}")
        print(f"训练集 F1 分数: {train_f1:.4f}")
        print(f"测试集 F1 分数: {test_f1:.4f}")
        print(f"训练集 精确率: {train_precision:.4f}")
        print(f"测试集 精确率: {test_precision:.4f}")
        print(f"训练集 召回率: {train_recall:.4f}")
        print(f"测试集 召回率: {test_recall:.4f}")
        print('***********************************************************')
        print()

        # 将当前信噪比下的指标添加到 DataFrame 中
        metrics_df = metrics_df.append({
            'SNR': dB,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'roc_auc_score': test_auc,
            'f1_score': test_f1,
            'precision_score': test_precision,
            'recall_score': test_recall
        }, ignore_index=True)

        # 记录每个信噪比下的损失
        loss_row = {
            'SNR': dB,
            **{f'mlp_loss_{i}': loss for i, loss in enumerate(mlp_losses)},
            **{f'cnn_loss_{i}': loss for i, loss in enumerate(cnn_losses)},
            **{f'aco_loss_{i}': loss for i, loss in enumerate(aco_losses)}
        }
        loss_df = loss_df.append(loss_row, ignore_index=True)

    metrics_df.to_csv('../output_data4/my_idea_ant_metrics.csv', index=False)
    loss_df.to_csv('../output_data4/my_idea_ant_combined_model_losses.csv', index=False)