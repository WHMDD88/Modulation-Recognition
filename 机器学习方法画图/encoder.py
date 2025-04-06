import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib.ticker import PercentFormatter

# 解决中文显示和负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(file_path, snr):
    data_df = pd.read_csv(file_path, encoding='UTF-8')
    dB_data_df = data_df[data_df['SNR'] == snr]
    no_info_col = ['index', 'Modulation', 'SNR','mean_amp','fre_band','var_amp','center_fre']
    X_data = dB_data_df.drop(columns=no_info_col)
    return X_data, dB_data_df


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

def get_encoded_features(data,encoding_dim = 3):
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
    num_epochs = 10
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

def perform_kmeans(X_data, n_clusters=11, max_iter=1000, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=max_iter, random_state=random_state)
    kmeans.fit(X_data)
    clusters = kmeans.labels_
    return clusters


def plot_2d_scatter(data, clusters,snr,train_num):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        data.values[:, 0],
        data.values[:, 1],
        c=clusters,
        cmap='viridis',
        s=80,
        alpha=0.8,
        edgecolor='k'
    )
    plt.xlabel('AE feature1', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    plt.ylabel('AE feature2', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    cbar = plt.colorbar(scatter, label='Cluster Label')
    cbar.ax.tick_params(labelsize=12)
    plt.grid(linestyle='--', alpha=0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('pictures\\enconder_'+str(snr)+'dB_'+str(train_num)+'_2D.svg')


def plot_3d_scatter(data, clusters,snr,train_num):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        data.values[:, 0],
        data.values[:, 1],
        data.values[:, 2],
        c=clusters,
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolor='k'
    )
    ax.set_xlabel('AE feature1', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.set_ylabel('AE feature2', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.set_zlabel('AE feature3', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.view_init(elev=30, azim=45)
    ax.grid(linestyle='--', alpha=0.4)
    cbar = plt.colorbar(scatter, label='Cluster Label', shrink=0.7)
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    plt.savefig('pictures\\enconder_'+str(snr)+'dB_'+str(train_num)+'_3D.svg')

def analyze_clusters(dB_data_df):
    mudulations = list(set(dB_data_df['Modulation']))
    for mod in mudulations:
        print("调制方式为:", mod)
        data_df2 = dB_data_df[dB_data_df['Modulation'] == mod]
        value_counts = data_df2['cluster_label'].value_counts()
        most_common_value = value_counts.idxmax()
        most_common_count = value_counts.max()
        print(f"数量最多的值是: {most_common_value}，出现次数为: {most_common_count}")
        print()
    print('*************************************')
    for label in range(11):
        print("label:", label)
        data_df2 = dB_data_df[dB_data_df['cluster_label'] == label]
        value_counts = data_df2['Modulation'].value_counts()
        most_common_value = value_counts.idxmax()
        most_common_count = value_counts.max()
        print(f"数量最多的值是: {most_common_value}，出现次数为: {most_common_count}")
        print()



if __name__ == '__main__':
    file_path = '../output_data1/nor_all_feas.csv'
    snr = 0
    train_num=5
    X_data, dB_data_df = load_and_preprocess_data(file_path, snr)

    data = get_encoded_features(X_data.values)
    X_encoder=pd.DataFrame(data)
    print(X_encoder.info())
    clusters = perform_kmeans(X_encoder)

    # 新增：计算聚类评价指标
    true_labels = dB_data_df['Modulation'].values
    silhouette_score_val = metrics.silhouette_score(X_data, clusters)
    calinski_harabasz_score_val = metrics.calinski_harabasz_score(X_data, clusters)
    print(f"轮廓系数(Silhouette Score): {silhouette_score_val:.4f}")
    print(f"Calinski-Harabasz指数: {calinski_harabasz_score_val:.4f}")

    plot_2d_scatter(X_encoder, clusters,snr,train_num)
    plot_3d_scatter(X_encoder, clusters,snr,train_num)
    plt.show()
    dB_data_df['cluster_label'] = clusters
    print(dB_data_df.info())
