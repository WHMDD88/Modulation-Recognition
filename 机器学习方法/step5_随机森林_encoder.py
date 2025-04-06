import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def create_feture_sel_model(X, Y):
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X, Y)
    return model

def get_feas_impoortance(X, Y, top_k):
    # X,Y 都是dataframe结构
    #print("select feature...")
    select_feature_model = create_feture_sel_model(X, Y)
    importances = select_feature_model.feature_importances_

    cols = X.columns
    pd1 = pd.DataFrame(importances, cols, columns=['importance'])
    pd1.sort_values(by='importance', ascending=True).plot(kind='barh', figsize=(4, 10))
    plt.grid()
    #plt.show()
    # 用置信区间来计算相对重要性的估计值

    std = np.std([i.feature_importances_ for i in select_feature_model.estimators_], axis=0)
    feat_with_importance = pd.Series(importances, X.columns)
    fig, ax = plt.subplots(figsize=(12, 5))
    feat_with_importance.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decraese in impurity")
    plt.grid()
    #plt.show()
    #print("feature selecting finish")
    # 按重要性的降序来选着前K个特征
    X_result = X.iloc[:, select_feature_model.feature_importances_.argsort()[::-1][:top_k]]
    #print(f'select {top_k} features')
    select_feas=X_result.columns.tolist()
    return select_feas

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_encoded_features(data,encoding_dim = 4):
    """
    :param data: dataframe的values
    :return:
    """
    # 转换为PyTorch张量
    data_tensor = torch.tensor(data, dtype=torch.float32)
    input_dim = data.shape[1]
    autoencoder = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()  # 均方误差
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
    num_epochs = 50
    for epoch in range(num_epochs):
        outputs = autoencoder(data_tensor)
        loss = criterion(outputs, data_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    encoder = nn.Sequential(*list(autoencoder.children())[:-1])
    with torch.no_grad():
        encoded_features = encoder(data_tensor).numpy()
    return encoded_features

if __name__ == '__main__':
    data_df = pd.read_csv('../output_data1/nor_all_feas.csv', encoding='UTF-8')
    SNR_list = list(i for i in range(-20, 19, 2))
    metrics_df = pd.DataFrame(
        columns=['SNR', 'train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score'])

    # 定义标签映射字典
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

    for select_SNR in SNR_list:
        print("信噪比:", select_SNR)
        sampled_df = data_df[data_df['SNR'] == select_SNR]

        X_df = sampled_df.drop(columns=['index', 'Modulation', 'SNR']).reset_index(drop=True)
        Y_df = sampled_df[['Modulation']].reset_index(drop=True)

        # 将标签转换为数值类型
        # Y_df['Modulation'] = Y_df['Modulation'].map(mapping_dict)

        top_k = 12
        selected_features = get_feas_impoortance(X_df, Y_df['Modulation'], top_k)
        print("选择的特征：")
        print(selected_features)

        X_data = X_df[selected_features]

        # 自编码器
        X_encoder = get_encoded_features(X_data.values, encoding_dim=8)
        X_encoder = pd.DataFrame(X_encoder)

        X_Y = X_encoder.copy()
        X_Y['Modulation'] = Y_df['Modulation']

        X_train, X_valid, y_train, y_valid = train_test_split(X_encoder, Y_df['Modulation'], test_size=0.08,
                                                              random_state=42)

        XY_df = X_Y.sample(frac=0.22)
        X1_df = XY_df.drop(columns=['Modulation'])
        Y1_df = XY_df['Modulation']

        X_test = pd.concat([X_valid, X1_df], axis=0, ignore_index=True)
        y_test = pd.concat([y_valid, Y1_df], axis=0, ignore_index=True)

        clf =RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        y_hat = clf.predict(X_train)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print('**********************************************************************')
        print()
        test_acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_hat)
        print("test_acc is {:.5}".format(test_acc))
        print("train_acc is {:.5}".format(train_acc))
        print(classification_report(y_test, y_pred, digits=3))
        auc_value = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
        print('roc_auc_score is {:.5}'.format(auc_value))
        f1 = f1_score(y_test, y_pred, average='micro')
        print('f1_score is {:.5}'.format(f1))
        precision = precision_score(y_test, y_pred, average='micro')
        print('precision score is {:.5}'.format(precision))
        recall = recall_score(y_test, y_pred, average='micro')
        print('recall score is {:.5}'.format(recall))
        print("********************************************************")
        print()

        # 将当前信噪比下的指标添加到 DataFrame 中
        metrics_df = metrics_df.append({
            'SNR': select_SNR,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'roc_auc_score': auc_value,
            'f1_score': f1,
            'precision_score': precision,
            'recall_score': recall
        }, ignore_index=True)

    #metrics_df.to_csv('../output_data3/RandomForest_encoder.csv', index=False)

