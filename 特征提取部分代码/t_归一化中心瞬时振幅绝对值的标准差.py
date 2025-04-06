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

def calculate_std(sample):
    amplitude = np.abs(sample)
    Ns = len(amplitude)
    ma = np.mean(amplitude)
    an = amplitude / ma
    acn = an - 1
    sum_acn_squared = np.sum(np.square(acn))
    mean_acn = np.mean(acn)
    std = np.sqrt((1 / Ns) * sum_acn_squared - np.square(mean_acn))
    return [std]

def get_data(data_dict):
    result_dict = {}
    for key, samples in data_dict.items():
        sample_results = []
        for sample in samples:
            std_value = calculate_std(sample)
            sample_results.append(std_value)
        result_dict[key] = sample_results
    return result_dict

if __name__ == '__main__':
    settings_random_seed()
    data_dict = load_data()
    # 合并
    merged_dict = merge_IQ_data(data_dict)
    # print(merged_dict[('QPSK', 2)])
    # print(type(merged_dict[('QPSK', 2)]))

    # 归一化
    # 创建新字典存储归一化后的数据
    normalized_dict = {}
    for key, value in merged_dict.items():
        normalized_value = normalize_complex_samples(value)
        normalized_dict[key] = normalized_value
    # print(normalized_dict[('QPSK', 2)])

    """result_dict=get_data(normalized_dict)


    with open('../output_data5/std.pickle', 'wb') as f:
        pickle.dump(result_dict, f)"""

    with open('../output_data/std.pickle','rb') as f:
        fea_dict=pickle.load(f)
    print(fea_dict)
