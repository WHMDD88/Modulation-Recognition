import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_hot_pic(df, fea_cols, target=None):
    # 数据预处理：映射目标值
    mapping_dict = {
        'AM-DSB': 0,
        'AM-SSB': 1,
        'WBFM': 2,
        'BPSK': 3,
        'QPSK': 4,
        '8PSK': 5,
        'CPFSK': 6,
        'GFSK': 7,
        'PAM4': 8,
        'QAM16': 9,
        'QAM64': 10
    }
    if target:
        df[target] = df[target].map(mapping_dict)

    # 提取数据并计算相关性
    X_data = df[fea_cols]
    corr_matrix = X_data.corr()

    # 绘制热力图
    plt.figure(figsize=(12, 8))  # 设置画布尺寸
    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # 发散型配色，区分正负相关

    # 绘制热力图并添加详细参数
    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        annot=True,  # 显示数值
        fmt=".2f",  # 数值格式
        linewidths=0.5,  # 单元格边框宽度
        center=0,  # 颜色中心值
        annot_kws={"fontsize": 10},  # 数值字体大小
        cbar=True,  # 显示颜色条
        cbar_kws={"shrink": 0.8, "label": "相关系数"}  # 颜色条参数
    )

    # 优化坐标轴标签
    plt.xticks(rotation=45, ha='right')  # X轴标签旋转45度并右对齐
    plt.yticks(rotation=0)  # Y轴标签不旋转
    #plt.title("特征相关性热力图", fontsize=14, pad=20)  # 添加标题

    # 保存并显示图像
    plt.savefig('pictures\\特征相关性热力图2.svg')  # 提高dpi，紧凑布局
    plt.show()


if __name__ == '__main__':
    data_df = pd.read_csv('../output_data1/nor_all_feas.csv', encoding='UTF-8')
    print(data_df.info())

    # 处理特征列
    fea_cols = data_df.columns.tolist()
    fea_cols = [col for col in fea_cols if col not in ['index', 'SNR','Modulation']]
    print(fea_cols)

    target = 'Modulation'
    draw_hot_pic(data_df, fea_cols)