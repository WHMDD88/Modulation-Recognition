import numpy as np
import pandas as pd
import torch
from IPython import display
from d2l import torch as d2l
from torch import nn
from torch.utils import data
from d2l import torch as d2l
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, roc_curve, \
    ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from torch.nn import functional as F
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0, 0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0, 0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 分类精度
def accuracy(y_hat, y):
    """计算预测正确的数量
     函数首先检查y_hat的形状，如果它有多个维度并且第二个维度大于1，
     那么将y_hat转换为类别索引。接下来，比较y_hat和y的数据类型是否相同，
     如果相同，则计算匹配的数量。最后，返回匹配数量的浮点值"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    # 将 y_hat 转换为与 y 相同的数据类型，然后比较它们是否相等
    ret = float(cmp.type(y.dtype).sum())
    return ret

def evaluate_accuracy(net, data_iter):
    """计算精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # numel()方法获取张量中元素的总数。
    return metric[0] / metric[1]

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss(reduction='none')
    metric = Accumulator(3)
    epoch_losses = []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            optimizer.step()

            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

        l_train = loss(net(train_features), train_labels).mean()
        train_ls.append(l_train.item())

        if test_labels is not None:
            l_test = loss(net(test_features), test_labels).mean()
            test_ls.append(l_test.item())

        # 使用acc作为指标
        train_metrics = metric[0] / metric[2], metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = train_metrics
        epoch_losses.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': l_test.item() if test_labels is not None else None
        })
    return train_ls, test_ls, train_acc, test_acc, epoch_losses

def k_fold(net, k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, snr):
    """10折训练开始"""
    train_l_sum, valid_l_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    all_epoch_losses = []
    for i in range(k):
        # 得到每一折的训练集,验证集
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, X_train, y_train)
        # 训练num_epochs
        train_ls, valid_ls, train_acc, valid_acc, epoch_losses = train(net, X_train, y_train, X_valid, y_valid,
                                                                       num_epochs, learning_rate, weight_decay,
                                                                       batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        train_acc_sum += train_acc
        valid_acc_sum += valid_acc

        for loss_info in epoch_losses:
            loss_info['fold'] = i + 1
            loss_info['SNR'] = snr
        all_epoch_losses.extend(epoch_losses)

    return train_l_sum / k, valid_l_sum / k, train_acc_sum / k, valid_acc_sum / k, all_epoch_losses

def draw(trues, preds, proba):
    # 混淆矩阵
    cm = confusion_matrix(trues, preds)
    cm_diaply = ConfusionMatrixDisplay(cm)
    cm_diaply.plot()
    #plt.show()
    acc = accuracy_score(trues, preds)
    print("test_acc is {:.5}".format(acc))
    # AUC
    auc1 = roc_auc_score(trues, proba, multi_class='ovo')
    print('auc score is {:.5}'.format(auc1))
    # f1
    f1 = f1_score(trues, preds, average='micro')
    print('f1 score is {:.5}'.format(f1))
    precision = precision_score(trues, preds, average='micro', zero_division=0)
    print('precision score is {:.5}'.format(precision))
    recall = recall_score(trues, preds, average='micro')
    print('recall score is {:.5}'.format(recall))

    return acc, auc1, f1, precision, recall

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
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_df, test_size=0.3, random_state=42)
    train_features = torch.tensor(X_train.values, dtype=torch.float32)
    test_features = torch.tensor(X_test.values, dtype=torch.float32)
    train_labels = torch.tensor(y_train.values, dtype=torch.long)
    test_labels = torch.tensor(y_test.values, dtype=torch.long)
    return train_features, test_features, train_labels, test_labels

def train_and_evaluate(train_features, train_labels, test_features, test_labels, snr):
    num_inputs = train_features.shape[1]
    num_outputs = 11
    units_hidden = 64
    k, num_epochs, lr, batch_size, weight_decay = 5, 50, 0.01, 256, 0
    net = get_net(num_inputs, units_hidden, num_outputs, 0.5)

    # 训练模式
    net.train()
    train_l, valid_l, train_acc, valid_acc, all_epoch_losses = k_fold(net, k, train_features, train_labels, num_epochs, lr,
                                                                       weight_decay, batch_size, snr)
    # 修改此处，去掉 item() 方法调用
    print(f'{k}-折验证:平均train_acc: {train_acc:f}  平均test_acc: {valid_acc:f}')
    print("测试模型")
    print("train_acc:", train_acc)
    net.eval()
    with torch.no_grad():
        outputs = net(test_features)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    proba = probabilities.numpy()
    preds = pd.Series(predicted)
    test_acc, auc1, f1, precision, recall = draw(test_labels, preds, proba)
    return train_acc, test_acc, auc1, f1, precision, recall, all_epoch_losses

if __name__ == '__main__':
    try:
        df = pd.read_csv('../output_data1/nor_all_feas.csv', encoding='UTF-8')
    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
        exit(1)
    dB_list = [i for i in range(-20, 19, 2)]
    metrics_df = pd.DataFrame(
        columns=['SNR', 'train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score'])
    all_epoch_losses_list = []

    for dB in dB_list:
        print("信噪比:", dB)
        train_features, test_features, train_labels, test_labels = preprocess_data(df, dB)
        train_acc, test_acc, auc1, f1, precision, recall, epoch_losses = train_and_evaluate(train_features, train_labels, test_features, test_labels, dB)
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

        all_epoch_losses_list.extend(epoch_losses)

    #metrics_df.to_csv('../output_data4/MLP_epoch100.csv', index=False)

    # 将每个epoch的损失信息保存为CSV文件
    epoch_losses_df = pd.DataFrame(all_epoch_losses_list)
    #epoch_losses_df.to_csv('../output_data4/MLP_epoch100_losses.csv', index=False)