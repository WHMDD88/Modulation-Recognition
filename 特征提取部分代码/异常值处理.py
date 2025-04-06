import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def load_data():
    filename = '../dataset/RML2016.10a_dict.pkl'
    try:
        with open(filename, 'rb') as open_file:
            Xd = pickle.load(open_file, encoding='latin1')
        print(type(Xd))
        snrs = sorted({key[1] for key in Xd.keys()})
        mods = sorted({key[0] for key in Xd.keys()})
        return Xd, mods
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到。")
        return None, None


def calculate_stats(Xd, mods):
    """
    按调制方式和信噪比分组计算I/Q两路信号的均值和标准差
    """
    stats = {}
    for mod in mods:
        for snr in range(-20, 20, 2):
            key = (mod, snr)
            if key in Xd:
                samples = Xd[key]  # (1000, 2, 128)
                I = samples[:, 0, :]  # (1000, 128)
                Q = samples[:, 1, :]  # (1000, 128)
                mean_I = np.mean(I, axis=0)  # (128,)
                std_I = np.std(I, axis=0)  # (128,)
                mean_Q = np.mean(Q, axis=0)  # (128,)
                std_Q = np.std(Q, axis=0)  # (128,)
                stats[key] = {
                    'mean_I': mean_I,
                    'std_I': std_I,
                    'mean_Q': mean_Q,
                    'std_Q': std_Q
                }
    return stats

def detect_outliers(Xd, stats, k=3):
    """
    检测异常值
    """
    outlier_masks = {}
    for key in Xd:
        samples = Xd[key]
        I = samples[:, 0, :]
        Q = samples[:, 1, :]
        mean_I = stats[key]['mean_I']
        std_I = stats[key]['std_I']
        mean_Q = stats[key]['mean_Q']
        std_Q = stats[key]['std_Q']
        mask_I = np.abs(I - mean_I) > k * std_I
        mask_Q = np.abs(Q - mean_Q) > k * std_Q
        outlier_mask = np.logical_or(mask_I, mask_Q)
        outlier_masks[key] = outlier_mask
    return outlier_masks

# 假设已经有 stats 字典，包含每组的均值和标准差
def plot_mean_std(stats):
    for key in stats:
        mod, snr = key
        mean_I = stats[key]['mean_I']
        std_I = stats[key]['std_I']
        mean_Q = stats[key]['mean_Q']
        std_Q = stats[key]['std_Q']

        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        plt.plot(mean_I, label='Mean I',color='purple')
        plt.fill_between(range(len(mean_I)), mean_I - std_I, mean_I + std_I, alpha=0.2, label='Std I',color='pink')
        #plt.title(f'{mod} at {snr}dB - I路',fontdict={'fontname':'SimSun'},fontsize=16)
        plt.xlabel('采样点',fontdict={'fontname':'SimSun'},fontsize=12)
        plt.ylabel('')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(mean_Q, label='Mean Q')
        plt.fill_between(range(len(mean_Q)), mean_Q - std_Q, mean_Q + std_Q, alpha=0.2, label='Std Q')
        #plt.title(f'{mod} at {snr}dB - Q路',fontdict={'fontname':'SimSun'},fontsize=16)
        plt.xlabel('采样点',fontdict={'fontname':'SimSun'},fontsize=12)
        plt.ylabel('')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'outerdetect\\mean_std\\{mod}_{snr}dB.svg')
        #plt.show()

def plot_constellation(Xd, key, stats):
    samples = Xd[key]
    I = samples[:, 0, :].flatten()
    Q = samples[:, 1, :].flatten()
    mean_I = stats[key]['mean_I'].mean()
    std_I = stats[key]['std_I'].mean()
    mean_Q = stats[key]['mean_Q'].mean()
    std_Q = stats[key]['std_Q'].mean()
    # 判断异常点
    outlier_mask_I = np.abs(I - mean_I) > 3 * std_I
    outlier_mask_Q = np.abs(Q - mean_Q) > 3 * std_Q
    outlier_mask = np.logical_or(outlier_mask_I, outlier_mask_Q)
    # 正常点
    normal_I = I[~outlier_mask]
    normal_Q = Q[~outlier_mask]
    # 异常点
    outlier_I = I[outlier_mask]
    outlier_Q = Q[outlier_mask]
    plt.figure(figsize=(6, 6))
    # 绘制正常点
    plt.scatter(normal_I, normal_Q, s=1, color='#77dd77', label='正常点')
    # 绘制异常点
    plt.scatter(outlier_I, outlier_Q, s=8, color='#d62728', label='异常点')
    #plt.title(f'Constellation Diagram - {key[0]} at {key[1]}dB')
    plt.xlabel('I',fontdict={'fontname':'Times New Roman'},fontsize=12)
    plt.ylabel('Q',fontdict={'fontname':'Times New Roman'},fontsize=12)
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(f'outerdetect\\IQ\\{key[0]}_{key[1]}dB.png',dpi=800)
    #plt.show()

def plot_constellation3(Xd, key, stats):
    samples = Xd[key]
    I = samples[:, 0, :].flatten()
    Q = samples[:, 1, :].flatten()

    mean_I = stats[key]['mean_I'].mean()
    std_I = stats[key]['std_I'].mean()
    mean_Q = stats[key]['mean_Q'].mean()
    std_Q = stats[key]['std_Q'].mean()

    # 判断异常点
    outlier_mask_I = np.abs(I - mean_I) > 3 * std_I
    outlier_mask_Q = np.abs(Q - mean_Q) > 3 * std_Q
    outlier_mask = np.logical_or(outlier_mask_I, outlier_mask_Q)

    # 正常点
    normal_I = I[~outlier_mask]
    normal_Q = Q[~outlier_mask]

    # 异常点
    outlier_I = I[outlier_mask]
    outlier_Q = Q[outlier_mask]

    plt.figure(figsize=(6, 6))

    # 计算二维直方图
    hist, xedges, yedges = np.histogram2d(normal_I, normal_Q, bins=50)
    # 获取每个点所在的直方图区间
    x_indices = np.digitize(normal_I, xedges)
    y_indices = np.digitize(normal_Q, yedges)
    # 避免索引越界
    x_indices = np.clip(x_indices - 1, 0, len(xedges) - 2)
    y_indices = np.clip(y_indices - 1, 0, len(yedges) - 2)
    # 获取每个点对应的密度值
    densities = hist[x_indices, y_indices]
    # 对密度进行归一化处理，使颜色映射到 [0, 1] 范围
    norm_densities = (densities - densities.min()) / (
        densities.max() - densities.min()
    )
    # 绘制正常点，根据密度设置颜色
    plt.scatter(
        normal_I, normal_Q, s=2, c=norm_densities, cmap='winter'
    )
    # 绘制异常点
    plt.scatter(outlier_I, outlier_Q, s=8, color="#d62728", label="异常点")
    plt.xlabel("I", fontdict={"fontname": "Times New Roman"}, fontsize=12)
    plt.ylabel("Q", fontdict={"fontname": "Times New Roman"}, fontsize=12)
    plt.grid()
    plt.legend(loc="upper right")
    #plt.savefig(f"outerdetect\\IQ\\{key[0]}_{key[1]}dB_密度图.png", dpi=800)
    plt.show()

def plot_outlier_histogram(outlier_masks):
    for key in outlier_masks:
        outlier_count_per_sample = np.sum(outlier_masks[key], axis=1)
        plt.figure(figsize=(6, 4))
        plt.hist(outlier_count_per_sample, bins=20,color='#1abc9c')
        #plt.title(f'异常值数量分布 - {key[0]} 在 {key[1]}dB')
        plt.xlabel('区间',fontdict={'fontname':'SimSun'},fontsize=12)
        plt.ylabel('频数',fontdict={'fontname':'SimSun'},fontsize=12)
        plt.grid(True)
        plt.savefig(f'outerdetect\\hist\\{key[0]}_{key[1]}dB.svg')
        #plt.show()

def filter_and_sort_stats(stats, target_mod, target_snrs):
    """
    筛选指定调制方式和指定信噪比的统计信息，并按信噪比从小到大排序。
    :param stats: 所有调制方式和信噪比的统计信息字典
    :param target_mod: 目标调制方式
    :param target_snrs: 目标信噪比列表
    :return: 排序后的指定调制方式和信噪比的统计信息字典
    """
    filtered_stats = {key: stats[key] for key in stats if key[0] == target_mod and key[1] in target_snrs}
    sorted_keys = sorted(filtered_stats.keys(), key=lambda x: x[1])
    return {key: filtered_stats[key] for key in sorted_keys}

def filter_and_sort_keys(data_dict, target_mod, target_snrs):
    """
    筛选指定调制方式和指定信噪比的键，并按信噪比从小到大排序。
    :param data_dict: 数据字典
    :param target_mod: 目标调制方式
    :param target_snrs: 目标信噪比列表
    :return: 排序后的指定调制方式和信噪比的键列表
    """
    keys = [key for key in data_dict.keys() if key[0] == target_mod and key[1] in target_snrs]
    keys.sort(key=lambda x: x[1])
    return keys

def main():
    Xd, mods = load_data()
    if Xd is not None and mods is not None:
        stats = calculate_stats(Xd, mods)
        outlier_masks = detect_outliers(Xd, stats)
        target_snrs = [-16, -8, 0, 8, 16]
        target_mod = 'QAM64'

        # 绘制 QAM64 在指定信噪比下按信噪比排序的均值和标准差曲线
        """qam64_stats = filter_and_sort_stats(stats, target_mod, target_snrs)
        plot_mean_std(qam64_stats)

        # 绘制 QAM64 在指定信噪比下的星座图
        qam64_keys = filter_and_sort_keys(Xd, target_mod, target_snrs)
        for key in qam64_keys:
            plot_constellation3(Xd, key, stats)

        # 绘制 QAM64 在指定信噪比下的异常值分布直方图
        qam64_outlier_masks = filter_and_sort_stats(outlier_masks, target_mod, target_snrs)
        plot_outlier_histogram(qam64_outlier_masks)"""

if __name__ == "__main__":
    main()
