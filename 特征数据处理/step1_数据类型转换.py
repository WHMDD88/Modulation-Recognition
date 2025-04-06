import pickle

import pandas as pd
import numpy as np
from sympy.codegen.fnodes import merge


def transform_data(data_dict,feaname_columns):
    # 初始化空列表，用于存储最终的数据
    data = []
    for (modulation, snr), samples in data_dict.items():
        for sample in samples:
            row = [modulation, snr] + sample
            data.append(row)
    columns = ['Modulation', 'SNR'] + feaname_columns
    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == '__main__':

    #读取信息熵特征
    #奇异谱香农熵、奇异谱指数熵、功率谱香农熵、功率谱指数熵
    #Singular Spectrum Shannon Entropy、Singular Spectrum Index Entropy、Power Spectrum Shannon Entropy、Power Spectrum Index Entropy
    """Entropy_feas_names=['SSSE','SSIE','PSSE','PEIE']
    with open('../output_data5/Entropy_fea_dict.pickle','rb') as f:
        Entropy_fea_dict=pickle.load(f)

    Entropy_fea_df=transform_data(Entropy_fea_dict,Entropy_feas_names)
    print("1熵特征")
    print(Entropy_fea_df.info())
    print(Entropy_fea_df.head(5))
    print("*******************************************************************")
    print()
    Entropy_fea_df.to_csv('../output_data1/Entropy_fea_df.csv',index=True,index_label='index')


    #读取循环谱特征
    #峰值振幅、峰值数量、平均峰值间距、平均振幅、振幅方差、振幅偏度、振幅峰度、频带、中心频率
    #peak_amplitude、peak_count、average_peak_spacing、
    # mean_amplitude、variance_amplitude、skewness_amplitude、kurtosis_amplitude、
    #frequency_band、center_frequency
    Cyc_spectrum_names=['peak_amp','peak_cout','avg_peak_spa',
                        'mean_amp','var_amp','ske_amp','kur_amp',
                        'fre_band','center_fre']
    with open('../output_data5/cyclic_spectrum_feature_dict.pickle','rb') as f:
        Cyc_spectrum_fea_dict=pickle.load(f)
    Cyc_spectrum_fea_df=transform_data(Cyc_spectrum_fea_dict,Cyc_spectrum_names)

    print("2循环谱特征")
    print(Cyc_spectrum_fea_df.info())
    print(Cyc_spectrum_fea_df.head(5))
    print("*******************************************************************")
    print()
    Cyc_spectrum_fea_df.to_csv('../output_data1/Cyc_spectrum_fea_df.csv',index=True,index_label='index')
    

    #读取归一化中心瞬时振幅功率密度最大值
    # The maximum value of the normalized central instantaneous amplitude power density
    # (MVNCIAPD)
    MVNCIAPD_names=['max_pow_den']
    with open('../output_data5/power_density.pickle', 'rb') as f:
        MVNCIAPD_fea_dict = pickle.load(f)
    MVNCIAPD_fea_df = transform_data(MVNCIAPD_fea_dict, MVNCIAPD_names)

    print("3归一化中心瞬时振幅功率密度最大值")
    print(MVNCIAPD_fea_df.info())
    print(MVNCIAPD_fea_df.head(5))
    print("*******************************************************************")
    print()
    MVNCIAPD_fea_df.to_csv('../output_data1/MVNCIAPD_fea_df.csv',index=True,index_label='index')

    # 读取归一化中心瞬时振幅绝对值的标准差(std)
    Std_names = ['std']
    with open('../output_data5/std.pickle', 'rb') as f:
        Std_fea_dict = pickle.load(f)
    Std_fea_df = transform_data(Std_fea_dict, Std_names)
    print("4归一化中心瞬时振幅绝对值的标准差")
    print(Std_fea_df.info())
    print(Std_fea_df.head(5))
    print("*******************************************************************")
    print()
    Std_fea_df.to_csv('../output_data1/Std_fea_df.csv',index=True,index_label='index')
    

    #合并
    df1=pd.read_excel('../output_data1/MVNCIAPD_fea_df.xlsx',engine='openpyxl')
    df2=pd.read_excel('../output_data1/Std_fea_df.xlsx',engine='openpyxl')
    df22=df2[['index','std']]
    merge_df=pd.merge(df1,df22,on=['index'])
    print(merge_df.info())
    print(merge_df.head(5))
    merge_df.to_excel('../output_data1/inst_amp_fea.xlsx',index=False)
    merge_df.to_csv('../output_data1/inst_amp_fea.csv',index=False)"""


    #合并所有的特征
    """target_cols=['Modulation','SNR']
    m_df1=pd.read_csv('../output_data1/Entropy_fea_df.csv',encoding='UTF-8')

    Modulations=m_df1[['Modulation']]
    snrs=m_df1[['SNR']]
    m_df1=m_df1.drop(columns=target_cols)
    m_df2 = pd.read_csv('../output_data1/Cyc_spectrum_fea_df.csv', encoding='UTF-8')
    m_df2 = m_df2.drop(columns=target_cols)
    m_df3 = pd.read_csv('../output_data1/inst_amp_fea.csv', encoding='UTF-8')
    m_df3 = m_df3.drop(columns=target_cols)
    merge_df1=pd.merge(m_df1,m_df2,on=['index'])
    print(merge_df1.info())
    merge_df2=pd.merge(merge_df1,m_df3,on=['index'])
    merge_df2['Modulation']=Modulations
    merge_df2['SNR']=snrs
    print(merge_df2.info())
    #merge_df2
    merge_df2.to_csv('../output_data1/all_feas.csv',index=False)"""









