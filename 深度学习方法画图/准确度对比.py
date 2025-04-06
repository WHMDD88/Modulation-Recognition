import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置字体以支持中文和负号显示
matplotlib.rc("font", family='Microsoft YaHei')
# 使用ggplot绘图风格，提供更美观的默认样式


def draw(col_name,y_name,ylim_value,line_styles):
    My_idea_df=pd.read_csv('../output_data5/DB_MCRF.csv',encoding='UTF-8')
    MLP_df=pd.read_csv('../output_data5/MLP_epoch100.csv',encoding='UTF-8')
    CNN_32_df=pd.read_csv('../output_data5/CNN32.csv',encoding='UTF-8')

    RF=pd.read_csv('../output_data5/RF.csv',encoding='UTF-8')
    Log=pd.read_csv('../output_data5/Log.csv',encoding='UTF-8')

    dfs = [My_idea_df,MLP_df,CNN_32_df,RF,Log]
    line_names = [
        'DB-MCRF',
        'MLP',
        'CNN',
        '堆叠法+RF',
        '堆叠法+LR'
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
    plt.savefig('pictures\\'+col_name+'对比.svg')
if __name__ == '__main__':
    col_names = ['train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
    y_list = ['Train Accuracy', 'Test Accuracy', 'AUC Score', 'F1 score', 'precision score', 'recall score']
    ylim_list = [(0, 1), (0.5, 1)]


    # 定义更美观的颜色和线条样式，添加标记点
    line_style1 = [
        {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
        {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'},
        {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^'},
        {'color': '#00CED1', 'linestyle': '--', 'marker': 'o'},
        {'color': '#DA70D6', 'linestyle': '-.', 'marker': 's'}
    ]
    draw(col_names[1],y_list[1],ylim_list[0],line_style1)
    plt.show()


