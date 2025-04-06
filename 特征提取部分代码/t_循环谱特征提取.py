import numpy as np
from scipy.signal import find_peaks

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

def cyclic_spectrum_estimation(x, N, L, alpha_max):
    """
    计算信号的循环谱
    :param x: 输入信号
    :param N: 信号长度
    :param L: 分段长度
    :param alpha_max: 最大循环频率
    :return: 循环频率向量 alpha，频率向量 f，循环谱 Sx_alpha_f
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


def extract_features(alpha, f, Sx_alpha_f):
    """
    从循环谱中提取峰值特征、统计特征和频带特征
    :param alpha: 循环频率向量
    :param f: 频率向量
    :param Sx_alpha_f: 循环谱
    :return: 包含各种特征的字典
    """
    abs_spectrum = np.abs(Sx_alpha_f)

    # 峰值特征
    peak_amplitude = np.max(abs_spectrum)
    threshold = 0.1 * peak_amplitude # 阈值
    flat_spectrum = abs_spectrum.flatten()
    peaks, _ = find_peaks(flat_spectrum, height=threshold)
    peak_count = len(peaks)
    if peak_count > 1:
        peak_positions = np.unravel_index(peaks, abs_spectrum.shape)
        peak_frequencies = f[peak_positions[1]]
        sorted_peak_frequencies = np.sort(peak_frequencies)
        peak_spacings = np.diff(sorted_peak_frequencies)
        average_peak_spacing = np.mean(peak_spacings) #  前两个峰值的间距？？？？
    else:
        average_peak_spacing = 0

    # 统计特征
    mean_amplitude = np.mean(abs_spectrum)
    variance_amplitude = np.var(abs_spectrum)
    skewness_amplitude = np.mean(((abs_spectrum - mean_amplitude) / np.sqrt(variance_amplitude)) ** 3)
    kurtosis_amplitude = np.mean(((abs_spectrum - mean_amplitude) / np.sqrt(variance_amplitude)) ** 4) - 3

    # 频带特征
    non_zero_indices = np.where(abs_spectrum > threshold)
    frequency_band = np.max(f[non_zero_indices[1]]) - np.min(f[non_zero_indices[1]])

    #最大频率和最小频率的均值？？？？
    center_frequency = (np.max(f[non_zero_indices[1]]) + np.min(f[non_zero_indices[1]])) / 2

    features = {
        'peak_amplitude': peak_amplitude,
        'peak_count': peak_count,
        'average_peak_spacing': average_peak_spacing,
        'mean_amplitude': mean_amplitude,
        'variance_amplitude': variance_amplitude,
        'skewness_amplitude': skewness_amplitude,
        'kurtosis_amplitude': kurtosis_amplitude,
        'frequency_band': frequency_band,
        'center_frequency': center_frequency
    }
    features_list=[peak_amplitude,peak_count,average_peak_spacing,
                   mean_amplitude,variance_amplitude,skewness_amplitude,kurtosis_amplitude,
                   frequency_band,center_frequency]
    return features_list


def cyclic_spectrum_feature(data_dict):
    # 循环谱计算参数
    L = 32
    alpha_max = 10
    # 存储所有样本的特征
    all_samples_features = {}
    # 遍历字典中的每个键值对
    for key, samples in data_dict.items():
        print("进行到:",key)
        samples_features = []
        for sample in samples:
            N = len(sample)
            alpha, f, Sx_alpha_f = cyclic_spectrum_estimation(sample, N, L, alpha_max)
            sample_features = extract_features(alpha, f, Sx_alpha_f)
            samples_features.append(sample_features)
        all_samples_features[key] = samples_features
    print("finsh")
    return all_samples_features


if __name__ == '__main__':

    """settings_random_seed()
    data_dict = load_data()
    # 合并
    merged_dict = merge_IQ_data(data_dict)
    # print(merged_dict[('QPSK', 2)])
    # print(type(merged_dict[('QPSK', 2)]))

    # 归一化
    normalized_dict = {}
    for key, value in merged_dict.items():
        normalized_value = normalize_complex_samples(value)
        normalized_dict[key] = normalized_value

    all_samples_features=cyclic_spectrum_feature(normalized_dict)
    print(all_samples_features)

    with open('../output_data5/cyclic_spectrum_feature_dict.pickle','wb') as f:
        pickle.dump(all_samples_features,f)"""
    with open('../output_data/cyclic_spectrum_feature_dict.pickle','rb') as f:
        fea_dict=pickle.load(f)

    print(fea_dict)





