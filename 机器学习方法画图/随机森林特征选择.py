import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_feture_sel_model(X, Y):
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X, Y)  # Y 现在是一维数组
    return model


def draw_importance(X, Y,snr,top_k=10):
    select_feature_model = create_feture_sel_model(X, Y)
    importances = select_feature_model.feature_importances_

    #重要性分数
    cols = X.columns
    pd1 = pd.DataFrame(importances, cols, columns=['importance'])
    pd1 = pd1.sort_values(by='importance', ascending=True)

    # 绘制水平柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = Normalize(vmin=pd1['importance'].min(), vmax=pd1['importance'].max())
    cmap = plt.get_cmap('Blues')
    colors = [cmap(norm(imp)) for imp in pd1['importance']]
    bars = ax.barh(pd1.index, pd1['importance'], color=colors, alpha=0.8, edgecolor='black')

    # 添加渐变色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='', ax=ax, shrink=0.6)

    ax.set_xlabel('重要性分数', fontsize=14,fontdict={'fontname':'SimSun'})
    #ax.set_ylabel('特征', fontsize=14,fontdict={'fontname':'SinSun'})
    #ax.set_title('特征重要性排序', fontsize=16,fontdict={'fontname':'SimSun'})
    ax.tick_params(axis='both', labelsize=12)
    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    #plt.savefig('pictures\\'+str(snr)+'dB_特征重要性排序.svg')
    plt.show()

    # 按重要性的降序来选着前K个特征
    X_result = X.iloc[:, select_feature_model.feature_importances_.argsort()[::-1][:top_k]]
    select_feas = X_result.columns.tolist()
    print('*******************************************************')
    print(f"{snr}dB下 选择的特征：")
    print(select_feas)
    print('******************************************************')
    print()


    std1 = np.std([i.feature_importances_ for i in select_feature_model.estimators_], axis=0)
    #std2=[]
    feat_with_importance = pd.Series(importances, X.columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    norm = Normalize(vmin=feat_with_importance.min(), vmax=feat_with_importance.max())
    cmap = plt.get_cmap('Greens')
    colors = [cmap(norm(imp)) for imp in feat_with_importance]

    bars = feat_with_importance.plot.bar(
        yerr=std1,
        ax=ax,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)
    )

    # ax.set_title("置信区间", fontsize=16,fontdict={'fontname':'SimSun'})
    #ax.set_ylabel("", fontsize=14, fontdict={'fontname': 'Times New Roman'})
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.grid(linestyle='--', alpha=0.6)

    # 添加右边渐变色条
    """sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        label='',
        orientation='vertical',  # 修改为垂直方向
        pad=0.05,  # 调整颜色条与图表的间距
        aspect=15,  # 调小aspect让颜色条更“胖”一点，但结合shrink缩小整体
        shrink=0.7,  # 新增，按比例缩小颜色条大小
    )
    #cbar.ax.set_ylabel('重要性值', labelpad=8, fontdict={'fontname': 'SimSun'})  # 调整标签位置

    plt.tight_layout()"""
    #plt.savefig('pictures\\'+str(snr)+'dB_置信区间.svg')
    plt.show()


if __name__ == '__main__':
    data_df = pd.read_csv('../output_data1/all_feas.csv', encoding='UTF-8')
    snrs=[i for i in range(-20,19,2) ]
    for select_SNR in snrs:
        sampled_df = data_df[data_df['SNR'] == select_SNR]
        X_df = sampled_df.drop(columns=['index', 'Modulation', 'SNR','ske_amp'])
        Y_df = sampled_df['Modulation']
        draw_importance(X_df, Y_df,select_SNR)