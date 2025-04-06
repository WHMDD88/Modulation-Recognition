import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


def plot_model_performance(model_name, col_name, y_name, ylim_value,line_styles):
    file_mapping = {
        'DecisionTree': ['../output_data3/DecisionTree.csv', '../output_data3/DecisionTree_pca.csv',
                         '../output_data3/DecisionTree_encoder.csv'],
        'LightGBM': ['../output_data3/LightGBM.csv', '../output_data3/LightGBM_pca.csv',
                     '../output_data3/LightGBM_encoder.csv'],
        'RandomForest': ['../output_data3/RandomForest.csv', '../output_data3/RandomForest_pca.csv',
                         '../output_data3/RandomForest_encoder.csv']
    }

    file_names = file_mapping[model_name]
    dfs = [pd.read_csv(file, encoding='UTF-8') for file in file_names]

    line_names = [
        model_name,
        f'{model_name} with PCA',
        f'{model_name} with AutoEncoder'
    ]

    plt.figure(figsize=(10, 6))
    for df, name, style in zip(dfs, line_names, line_styles):
        x = df['SNR']
        y = df[col_name]
        plt.plot(x, y,
                 label=name,
                 color=style['color'],
                 linestyle=style['linestyle'],
                 linewidth=2,
                 alpha=0.8,
                 marker=style['marker'],  # 添加标记点
                 markersize=6)  # 设置标记点大小

    # 统一设置图表样式
    plt.xlabel('SNR(dB)', fontdict={'fontname': 'Times New Roman', 'fontsize': 16})
    plt.ylabel(y_name, fontdict={'fontname': 'Times New Roman', 'fontsize': 16})
    plt.grid(linestyle='--', alpha=0.6)
    plt.ylim(ylim_value)

    # 优化图例
    plt.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.9,
        edgecolor='white',
        fontsize=14
    )

    plt.xticks([i for i in range(-20, 19, 2)], fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    #plt.savefig('pictures\\'+model_name+'_'+col_name+'_降维.svg')

if __name__ == '__main__':
    col_names = ['train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
    y_list = ['Train Accuracy', 'Test Accuracy', 'AUC Score', 'F1 score', 'precision score', 'recall score']
    ylim_list = [(0.1, 1), (0.5, 1)]


    # 定义更美观的颜色和线条样式，添加标记点
    line_style1 = [
        {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
        {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'},
        {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^'}
    ]
    line_style2 = [
        {'color': '#333333', 'linestyle': '-', 'marker': 'o'},
        {'color': '#00CED1', 'linestyle': '--', 'marker': 's'},
        {'color': '#DA70D6', 'linestyle': '-.', 'marker': '^'}
    ]

    models = ['DecisionTree', 'LightGBM', 'RandomForest']
    for model in models:
        plot_model_performance(
            model,
            col_names[2],
            y_list[2],
            ylim_list[1],
            line_style2
        )

    plt.show()