import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib
from joblib import Parallel, delayed

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
    filename = '../dataset/RML2016.10a_dict.pkl'
    with open(filename, 'rb') as open_file:
        Xd = pickle.load(open_file, encoding='latin1')
    return Xd

def normalize_complex_samples(samples):
    magnitudes = np.abs(samples)
    min_magnitudes = np.min(magnitudes, axis=1, keepdims=True)
    max_magnitudes = np.max(magnitudes, axis=1, keepdims=True)
    diff = max_magnitudes - min_magnitudes
    diff[diff == 0] = 1  # 避免除零错误
    normalized_magnitudes = (magnitudes - min_magnitudes) / diff
    phases = np.angle(samples)
    return normalized_magnitudes * np.exp(1j * phases)

def merge_IQ_data(dict_data):
    merged_dict = {}
    for key, value in dict_data.items():
        i_signal = value[:, 0, :]
        q_signal = value[:, 1, :]
        merged_signal = i_signal + 1j * q_signal
        merged_dict[key] = merged_signal
    return merged_dict

def cyclic_spectrum_estimation(x, N, L, alpha_max):
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

def process_signal(signal, L, alpha_max):
    N = len(signal)
    return cyclic_spectrum_estimation(signal, N, L, alpha_max)[2]

def cyclic_spectrum(data_dict):
    L = 32
    alpha_max = 10
    all_samples_feas = {}
    cout = 0
    num_cpus = 4  # 限制并行进程数量
    for key, samples in data_dict.items():
        cout += 1
        print(cout)
        print("进行到:", key)
        results = Parallel(n_jobs=num_cpus)(delayed(process_signal)(sample, L, alpha_max) for sample in samples)
        all_samples_feas[key] = results
    print("finsh")
    return all_samples_feas

if __name__ == '__main__':
    settings_random_seed()
    data_dict = load_data()
    merged_dict = merge_IQ_data(data_dict)
    normalized_dict = {}
    for key, value in merged_dict.items():
        normalized_value = normalize_complex_samples(value)
        normalized_dict[key] = normalized_value

    result_dict = cyclic_spectrum(normalized_dict)
    with open('../output_data3/cyclic_spectrum32.pickle', 'wb') as f:
        pickle.dump(result_dict, f)