import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib

# 设置字体以支持中文和负号显示
matplotlib.rc("font", family='Microsoft YaHei')

def plot_average(fea_dict):
    snrs = sorted({key[1] for key in fea_dict.keys()})
    mods = sorted({key[0] for key in fea_dict.keys()})

    plt.figure(figsize=(12, 8))
    for mod in mods:
        avg_y_max_values = []
        for snr in snrs:
            key = (mod, snr)
            if key in fea_dict:
                values = np.array(fea_dict[key])
                avg_y_max = np.mean(values)
                avg_y_max_values.append(avg_y_max)
        plt.plot(snrs, avg_y_max_values, marker='o', label=mod)

    plt.xlabel('SNR (dB)',fontsize=14,fontdict={'fontname':'Times New Roman'})
    plt.ylabel('归一化中心瞬时振幅模值的标准差',fontsize=14,fontdict={'fontname':'SimSun'})
    #plt.title('归一化中心瞬时振幅功率密度最大值的平均值')
    plt.legend()
    # 设置横坐标刻度
    plt.xticks([i for i in range(-20, 19, 2)], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.savefig('pictures\\归一化中心瞬时振幅模值的标准差.svg',dpi=800)
    plt.show()

if __name__ == '__main__':

    with open('../output_data/std.pickle', 'rb') as f:
        fea_dict = pickle.load(f)
    print(fea_dict)
    # 绘制图表
    plot_average(fea_dict)