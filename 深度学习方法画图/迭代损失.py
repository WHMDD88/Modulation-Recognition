import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import matplotlib
# 设置字体以支持中文和负号显示
matplotlib.rc("font", family='Microsoft YaHei')
# 使用ggplot绘图风格，提供更美观的默认样式
plt.style.use('ggplot')

if __name__ == '__main__':
    df=pd.read_csv('../output_data5/loss.csv',encoding='UTF-8')
    x=[i for i in range(1,501)]
    plt.figure(figsize=(12,6))
    plt.plot(x, df['mean'].values, color='#C71585', linewidth=2.88)
    plt.xlabel('迭代次数',fontsize=14,fontdict={'fontname':'SimSun'})
    plt.ylabel('Loss',fontsize=14,fontdict={'fontname':'Time News Roman'})
    plt.savefig('pictures/迭代损失.svg')
    plt.show()