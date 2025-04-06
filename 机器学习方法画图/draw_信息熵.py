import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置字体以支持中文和负号显示
matplotlib.rc("font", family='Microsoft YaHei')
# 使用ggplot绘图风格，提供更美观的默认样式
plt.style.use('ggplot')

def pre_draw(data_dict):
    feature_dicts = [{}, {}, {}, {}]
    # 整理数据
    for (modulation, snr), features in data_dict.items():
        for i in range(4):
            if modulation not in feature_dicts[i]:
                feature_dicts[i][modulation] = ([], [])  # 存储信噪比和对应的特征值
            feature_dicts[i][modulation][0].append(snr)
            feature_dicts[i][modulation][1].append(features[i])
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
    return feature_dicts

def draw(feature_dicts):
    # 定义不同调制方式的线条样式和颜色
    line_styles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 使用tab10颜色映射生成10种颜色
    number_chara = ['奇异谱香农熵', '奇异谱指数熵', '功率谱香农熵', '功率谱指数熵']
    # 绘制 4 个图
    for i in range(len(number_chara)):
        plt.figure(figsize=(10, 6))  # 设置图形大小
        for j, (modulation, (snrs, values)) in enumerate(feature_dicts[i].items()):
            color = colors[j % len(colors)]
            linestyle = line_styles[j % len(line_styles)]
            plt.plot(snrs, values, marker='o', label=modulation, color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
        plt.xlabel('SNR (dB)', fontsize=14,fontdict={'fontname':'Times New Roman'})
        plt.ylabel(number_chara[i], fontsize=14,fontdict={'fontname':'SimSun'})
        #plt.title(number_chara[i] + '随信噪比的变化', fontsize=16)
        # 优化图例布局
        plt.legend(loc='best', fontsize=12, frameon=True, framealpha=0.9, edgecolor='black')
        plt.grid(True, linestyle='--', alpha=0.6)  # 设置网格线样式
        # 设置横坐标刻度
        plt.xticks([i for i in range(-20, 19, 2)], fontsize=12)
        plt.yticks(fontsize=12)
        # 设置坐标轴范围
        plt.xlim(-21, 19)
        plt.tight_layout()  # 调整子图布局
        plt.savefig('pictures\\'+number_chara[i]+'.svg')
    plt.show()

if __name__ == '__main__':
    try:
        with open('../output_data/result_dict.pickle', 'rb') as f:
            loaded_dict = pickle.load(f)
    except FileNotFoundError:
        print("未找到指定的pickle文件，请检查文件路径。")
    except EOFError:
        print("pickle文件为空或损坏，请检查文件内容。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    else:
        # 创建新字典
        feature_dicts = pre_draw(loaded_dict)
        draw(feature_dicts)