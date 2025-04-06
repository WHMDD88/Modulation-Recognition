import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

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


def plot_iq_signals(data_dict, mods,snr,index_num):
    for mod in mods:
        # 选择一个信噪比（这里选择 0dB）和一个样本（这里选择第一个样本）
        key = (mod, snr)
        if key in data_dict:
            sample = data_dict[key][index_num]
            i_signal = sample[0]
            q_signal = sample[1]

            # 创建一个新的图形
            plt.figure(figsize=(8, 4))

            # 绘制 I 路信号
            plt.plot(i_signal, color='blue', label='I 路信号',marker='o', linestyle='-')
            # 绘制 Q 路信号
            plt.plot(q_signal, color='darkorange', label='Q 路信号',marker='s', linestyle='-')

            #plt.title(f'{mod} I 路和 Q 路信号')
            plt.xlabel('采样点',fontdict={'fontname':'SimSun'},fontsize=10)
            plt.ylabel('幅度',fontdict={'fontname':'SimSun'},fontsize=10)
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'pictures\\{snr}dB_{mod}_{index_num}_I路和Q路信号.svg')
            #plt.show()
        else:
            print(f"未找到 {key} 的数据。")


if __name__ == '__main__':
    data_dict, mods = load_data()
    snr=8
    index_num=288
    if data_dict and mods:
        plot_iq_signals(data_dict, mods,snr,index_num)
