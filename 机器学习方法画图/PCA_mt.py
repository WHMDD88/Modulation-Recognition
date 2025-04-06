import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics  # 新增：导入评价指标模块
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

def perform_pca(X_data, n_components=4):
    pca = PCA(n_components=n_components)
    data1 = pca.fit_transform(X_data)
    data = pd.DataFrame(data1)
    return data

def pca_analysis(X_data, snr):
    cov_mat = np.cov(X_data.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = list(zip(eig_vals, eig_vecs))
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    # 自动计算使方差达到 80% 的特征数量
    num_components = np.argmax(cum_var_exp >= 90) + 1
    print(f"使方差达到 80% 的特征数量为: {num_components}")
    # 绘制方差贡献率和累计方差贡献率图
    plt.figure(figsize=(10, 5))
    # 调整柱状图颜色为粉色
    bar_color = '#FFB6C1'
    plt.bar(range(len(var_exp)), var_exp, alpha=0.7, align="center", label='方差贡献率', color=bar_color)
    # 调整阶梯图颜色为更深的粉色
    step_color = '#FF69B4'
    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid', label='累计方差贡献率', color=step_color, linewidth=2)
    # 调整阈值线颜色和样式
    plt.axhline(y=90, color='r', linestyle='--', label='90% 方差阈值', linewidth=1.5)
    # 设置坐标轴标签和标题
    plt.ylabel("方差贡献率百分比",fontdict={'fontname':'SimSun'},fontsize=14)
    plt.xlabel("主成分数量",fontdict={'fontname':'SimSun'},fontsize=14)
    #plt.title("方差贡献率与累计方差贡献率",fontdict={'fontname':'SimSun'})
    # 设置坐标轴刻度为百分比形式
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    # 设置图例字体大小
    plt.legend(loc="best", fontsize=10)
    # 调整网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    # 保存图片
    plt.savefig('pictures\\PCA_'+str(snr)+'阶梯图.svg')
    plt.show()
    # 降维到满足 80% 方差的特征数量
    #pca = PCA(n_components=num_components)
    #data = pca.fit_transform(X_data)
    #data_df = pd.DataFrame(data)

def perform_kmeans(X_data, n_clusters=11, max_iter=1000, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=max_iter, random_state=random_state)
    kmeans.fit(X_data)
    clusters = kmeans.labels_
    return clusters


def plot_2d_scatter(data, clusters,snr):
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
    plt.xlabel('pca feature1', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    plt.ylabel('pca feature2', fontsize=14, fontdict={'fontname': 'Times New Roman'})
    cbar = plt.colorbar(scatter, label='Cluster Label')
    cbar.ax.tick_params(labelsize=12)
    plt.grid(linestyle='--', alpha=0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('pictures\\PCA_'+str(snr)+'_2D.svg')


def plot_3d_scatter(data, clusters,snr):
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
    ax.set_xlabel('pca feature1', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.set_ylabel('pca feature2', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.set_zlabel('pca feature3', fontsize=16, fontdict={'fontname': 'Times New Roman'})
    ax.view_init(elev=65, azim=45)
    ax.grid(linestyle='--', alpha=0.4)
    cbar = plt.colorbar(scatter, label='Cluster Label', shrink=0.7)
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    plt.savefig('pictures\\PCA_'+str(snr)+'_3D.svg')


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


if __name__ == "__main__":
    file_path = '../output_data1/nor_all_feas.csv'
    snr = 0
    X_data, dB_data_df = load_and_preprocess_data(file_path, snr)

    data = perform_pca(X_data)
    pca_analysis(X_data,snr)
    clusters = perform_kmeans(X_data)

    # 新增：计算聚类评价指标
    true_labels = dB_data_df['Modulation'].values
    silhouette_score_val = metrics.silhouette_score(X_data, clusters)
    calinski_harabasz_score_val = metrics.calinski_harabasz_score(X_data, clusters)
    print(f"轮廓系数(Silhouette Score): {silhouette_score_val:.4f}")
    print(f"Calinski-Harabasz指数: {calinski_harabasz_score_val:.4f}")

    plot_2d_scatter(data, clusters,snr)
    plot_3d_scatter(data, clusters,snr)
    plt.show()
    dB_data_df['cluster_label'] = clusters
    print(dB_data_df.info())
    #analyze_clusters(dB_data_df)