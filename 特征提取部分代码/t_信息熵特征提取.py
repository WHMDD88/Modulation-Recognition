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

# 定义你要进行的计算函数，这里以计算均值为例
def calculation(x_128):
    SVD_sn_entropy, SVD_idx_entropy = SVD_Entropy(x_128)
    power_sn_entropy, power_idx_entropy = power_entropy(x_128)

    return SVD_sn_entropy, SVD_idx_entropy, power_sn_entropy, power_idx_entropy


def SVD_Entropy(x_128):
    K = 40
    N = len(x_128)
    J = N - K + 1
    M = np.zeros((K, J), dtype=complex)  # 修改为复数类型
    for i in range(K):
        M[i, :] = x_128[i:i + J]
    U, s, Vt = np.linalg.svd(M)
    # s是奇异值向量，要得到Σ矩阵的形式，可进行如下处理
    Sigma = np.zeros((K, J), dtype=complex)  # 修改为复数类型
    Sigma[:min(K, J), :min(K, J)] = np.diag(s)
    sspectrum = s[s > 1e-10]
    # 使用一个小阈值（如1e-10）来过滤掉接近零的值，避免数值计算误差的影响
    # 计算奇异谱概率
    prob = sspectrum / np.sum(sspectrum)
    # 计算奇异谱香农熵
    shannon_entropy = -np.sum([p * np.log2(p) for p in prob if p > 0])
    # 计算奇异谱指数熵
    index_entropy = np.exp(np.sum([p * np.log2(1 - p) for p in prob if p > 0]))
    # print("奇异谱概率:", prob)
    # print("奇异谱香农熵:", shannon_entropy)
    # print("奇异谱指数熵:", index_entropy)
    return shannon_entropy, index_entropy

def power_entropy(x_128):
    """
    计算输入信号序列的功率谱香农熵和功率谱指数熵。
    参数:
    x_128 (np.ndarray): 输入的信号序列，可以是复数序列。
    返回:
    tuple: 包含功率谱香农熵和功率谱指数熵的元组。
    """
    # 检查输入是否为 numpy 数组
    if not isinstance(x_128, np.ndarray):
        x_128 = np.array(x_128)

    N = len(x_128)
    # 离散傅里叶变换
    y = np.fft.fft(x_128)
    # 计算功率谱值
    S = (1 / N) * np.abs(y) ** 2
    # 计算概率密度分布
    prob_density = S / np.sum(S)
    # 计算功率谱香农熵
    power_shannon_entropy = -np.sum([p * np.log2(p) for p in prob_density if p > 0])
    # 计算功率谱指数熵
    power_index_entropy = np.exp(np.sum([p * np.log2(1 - p) for p in prob_density if p > 0]))
    return power_shannon_entropy, power_index_entropy


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

def pre_data(dict_data):
    # 遍历原始字典的每个键值对
    new_dict={}
    for key, value in dict_data.items():
        print("进行到:",key)
        result_list=[]
        for sample in value:
            #print(sample.shape)
            SVD_sn_entropy, SVD_idx_entropy, power_sn_entropy, power_idx_entropy=calculation(sample)
            # 将四个计算结果组成一个列表
            result_tuple = [SVD_sn_entropy, SVD_idx_entropy, power_sn_entropy, power_idx_entropy]
            result_list.append(result_tuple)
        # 将结果列表作为新字典中对应键的值
        new_dict[key] = result_list
    return new_dict

def mean_data(data_dict):
    # 每个特征的均值  方便画图
    result_dict = {}
    for key, value_list in data_dict.items():
        # 获取元组的长度，假设所有元组长度相同
        tuple_length = len(value_list[0])
        means = []
        for i in range(tuple_length):
            # 提取每个元组的第 i 个元素
            elements = [tup[i] for tup in value_list]
            # 计算这些元素的均值
            mean = sum(elements) / len(elements)
            means.append(mean)
        result_dict[key] = means
    # print("result_dict:",result_dict)
    print("finish")
    return result_dict


def pre_draw(data_dict):
    feature_dicts = [{}, {}, {}, {}]
    # 整理数据
    for (modulation, snr), features in data_dict.items():
        for i in range(4):
            if modulation not in feature_dicts[i]:
                feature_dicts[i][modulation] = ([], [])  # 存储信噪比和对应的特征值
            feature_dicts[i][modulation][0].append(snr)
            feature_dicts[i][modulation][1].append(features[i])
    # print(feature_dicts)
    # 对每个调制方式的信噪比进行排序，并相应地排序特征值
    for feature_dict in feature_dicts:
        for modulation in feature_dict:
            snrs, values = feature_dict[modulation]
            # key 参数接受一个函数，该函数用于指定排序的依据。
            sorted_indices = sorted(range(len(snrs)), key=lambda k: snrs[k])
            feature_dict[modulation] = (
                [snrs[i] for i in sorted_indices],
                [values[i] for i in sorted_indices]
            )
    #print(feature_dicts)
    return feature_dicts

def draw(feature_dicts):
    # 绘制 4 个图
    number_chara=['奇异谱香农熵','奇异谱指数熵','功率谱香农熵','功率谱指数熵']
    for i in range(len(number_chara)):
        plt.figure()
        for modulation, (snrs, values) in feature_dicts[i].items():
            plt.plot(snrs, values, marker='o', label=modulation)
        plt.xlabel('信噪比')
        plt.ylabel(number_chara[i])
        plt.title(number_chara[i]+'随信噪比的变化')
        plt.legend(loc='best')
        plt.grid(True)
        # 设置横坐标刻度
        plt.xticks([i for i in range(-20, 19, 2)])
    plt.show()

if __name__ == '__main__':
    settings_random_seed()
    data_dict = load_data()
    #合并
    merged_dict=merge_IQ_data(data_dict)
    #print(merged_dict[('QPSK', 2)])
    #print(type(merged_dict[('QPSK', 2)]))

    # 归一化
    # 创建新字典存储归一化后的数据
    normalized_dict = {}
    for key, value in merged_dict.items():
        normalized_value = normalize_complex_samples(value)
        normalized_dict[key] = normalized_value
    #print(normalized_dict[('QPSK', 2)])

    #每一个样本的信息熵特征
    """dict1 = pre_data(normalized_dict)
    with open('../output_data5/Entropy_fea_dict.pickle','wb') as f:
        pickle.dump(dict1,f)"""
    with open('../output_data/Entropy_fea_dict.pickle','rb') as f:
        dict1=pickle.load(f)
    print(dict1[('QPSK',2)])
    #print(dict1)

    #画图
    #每个信噪比下的信息熵特征
    """dict2=mean_data(dict1)
    # 保存字典到文件
    with open('../output_data5/result_dict.pickle', 'wb') as f:
        pickle.dump(dict2, f)"""

    with open('../output_data/result_dict.pickle', 'rb') as f:
        loaded_dict = pickle.load(f)

    #print(loaded_dict)
    # 创建新字典
    feature_dicts = pre_draw(loaded_dict)
    draw(feature_dicts)





