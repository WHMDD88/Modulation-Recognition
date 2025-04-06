import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib
import os

from scipy import signal

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("设备类型：{}".format(device))
# 设置字体以支持中文和负号显示
matplotlib.rc("font", family='Microsoft YaHei')


def settings_random_seed():
    seed = 28
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    """
    RADIOML 2016.10a 数据集包含了从-20dB 到+18dB 总共 20 个信噪比（步长
    为 2）下的 11 种调制信号， 包括 AM-DSB、 AM-SSB 和 WBFM 三种模拟调制信
    号，以及 BPSK、 QPSK、 8PSK、 CPFSK、 GFSK、 PAM4、 QAM16 和 QAM64 八
    种数字调制信号。其中信号的中心频率为 200KHz，采样频率为 1Msamp/s，且每个
    信噪比下每种调制信号包含 1000 个信号。其中每个信号包含 IQ 两路数据，且每
    一路数据都包含有 128 个采样点。
    """
    filename = '../dataset/RML2016.10a_dict.pkl'
    open_file = open(filename, 'rb')
    Xd = pickle.load(open_file, encoding='latin1')
    print(type(Xd))
    # print(Xd[('QPSK', 2)].shape)
    # print(type(Xd[('QPSK', 2)]))
    # print("key:",list(Xd.keys()))
    # print("values:",Xd.values())
    snrs = sorted({key[1] for key in Xd.keys()})
    mods = sorted({key[0] for key in Xd.keys()})
    # print("信噪比种类:",snrs,'\n调制方式种类',mods)
    return Xd

def merge_IQ_data(dict_data):
    # 创建新字典存储合并后的数据
    merged_dict = {}
    for key, value in dict_data.items():
        # 提取 I 路和 Q 路信号
        i_signal = value[:, 0, :]
        q_signal = value[:, 1, :]
        # 合并 I/Q 两路信号为复数形式
        merged_signal = i_signal + 1j * q_signal
        merged_dict[key] = merged_signal
    return merged_dict


def normalize_complex_samples(samples):
    """
    对输入的复数样本进行归一化处理
    :param samples: 形状为 (1000, 128) 的 numpy.ndarray，元素为复数
    :return: 归一化后的 numpy.ndarray
    """
    normalized_samples = np.zeros_like(samples)
    for i in range(samples.shape[0]):
        sample = samples[i]
        # 计算样本的幅度
        magnitudes = np.abs(sample)
        min_magnitude = np.min(magnitudes)
        max_magnitude = np.max(magnitudes)
        if max_magnitude - min_magnitude > 0:
            # 计算归一化后的幅度
            normalized_magnitudes = (magnitudes - min_magnitude) / (max_magnitude - min_magnitude)
            # 保持相位不变，重新构建复数
            phases = np.angle(sample)
            normalized_sample = normalized_magnitudes * np.exp(1j * phases)
        else:
            # 处理最大值和最小值相等的情况，避免除零错误
            normalized_sample = sample
        normalized_samples[i] = normalized_sample
    return normalized_samples


def cyclic_spectrum_estimation(x, N, L, alpha_max):
    """
    循环谱估计函数
    :param x: 输入信号
    :param N: 信号长度
    :param L: 分段长度
    :param alpha_max: 最大循环频率
    :return: 循环频率向量 alpha, 频率向量 f, 循环谱 Sx_alpha_f
    """
    K = N // L
    Sx_alpha_f = np.zeros((2 * alpha_max + 1, L), dtype=complex)

    for alpha in range(-alpha_max, alpha_max + 1):
        Rx_alpha_tau = np.zeros(L, dtype=complex)
        for k in range(K):
            xk = x[k * L:(k + 1) * L]
            for tau in range(L):
                sum_term = 0
                for t in range(L - tau):
                    sum_term += xk[t + tau] * np.conj(xk[t]) * np.exp(-1j * 2 * np.pi * alpha * t / L)
                Rx_alpha_tau[tau] += sum_term
        Rx_alpha_tau /= K
        Sx_alpha_f[alpha + alpha_max, :] = np.fft.fft(Rx_alpha_tau)

    alpha = np.arange(-alpha_max, alpha_max + 1)
    f = np.fft.fftfreq(L)

    return alpha, f, Sx_alpha_f

def draw_Cyclic_Spectrum(data_dict):
    # 分段长度
    L = 16
    # 最大循环频率
    alpha_max = 10
    #print(data_dict.keys())
    modulation_types =  list(set([key[0] for key in data_dict.keys()]))
    # 提取调制方式到集合中去重
    print(modulation_types)
    snr = 8
    # 遍历不同的调制方式
    for modulation_type in modulation_types:
        # 获取对应调制方式和信噪比为 5dB 的数据
        key = (modulation_type, snr)
        data = data_dict[key]

        index_n=528
        sample = data[index_n]

        alpha, f, Sx_alpha_f = cyclic_spectrum_estimation(sample, len(sample), L, alpha_max)

        plt.figure(figsize=(10, 8))
        plt.imshow(
            np.abs(Sx_alpha_f),
            extent=[f.min(), f.max(), alpha.min(), alpha.max()],
            aspect='auto',
            origin='lower',
            cmap='cividis'  # 核心修改：指定颜色映射
        )
        plt.colorbar(label='Magnitude')
        plt.xlabel('Frequency (f)', fontdict={'fontname': 'Times New Roman', 'fontsize': 16})
        plt.ylabel('Cyclic Frequency (α)', fontdict={'fontname': 'Times New Roman', 'fontsize': 16})
        #plt.title(f'信噪比为{snr}dB下的{modulation_type}调制的循环谱')

        # 保存图片到 picture 目录
        picture_dir = "pictures16"
        filename = os.path.join(picture_dir, f'index_{index_n}_{modulation_type}_SNR_{snr}dB.svg')

        plt.savefig(filename, dpi=300)
        #plt.show()


if __name__ == '__main__':
    settings_random_seed()
    data_dict = load_data()
    # 合并
    merged_dict = merge_IQ_data(data_dict)

    # 归一化
    # 创建新字典存储归一化后的数据
    normalized_dict = {}
    for key, value in merged_dict.items():
        normalized_value = normalize_complex_samples(value)
        normalized_dict[key] = normalized_value
    # print(normalized_dict[('QPSK', 2)])

    draw_Cyclic_Spectrum(normalized_dict)