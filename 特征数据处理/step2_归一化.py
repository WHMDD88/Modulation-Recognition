import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def min_max_normalize(df, condition_col, condition_value, columns_to_nor):
    # 筛选满足条件的行
    condition = df[condition_col]==condition_value
    filtered_df = df[condition]

    for col in columns_to_nor:
        min_val = filtered_df[col].min()
        max_val = filtered_df[col].max()
        if max_val == min_val:
            # 处理最大值和最小值相等的情况，将该列数据都设为 0.5
            df.loc[condition, col] = 0.5
        else:
            scaler = MinMaxScaler()
            df.loc[condition, col] = scaler.fit_transform(filtered_df[[col]]).flatten()

    return df

if __name__ == '__main__':
    data_df = pd.read_csv("../output_data1/all_feas.csv")
    print(data_df.info())
    print(data_df.head(10))
    snrs=[i for i in range(-20,19,2)]
    print(snrs)
    col_nor_list=data_df.columns.tolist()
    col_nor_list.remove('index')
    col_nor_list.remove('Modulation')
    col_nor_list.remove('SNR')
    print(col_nor_list)
    for snr in snrs:
        data_df=min_max_normalize(data_df,'SNR',snr,col_nor_list)
    data_df.to_csv('../output_data1/nor_all_feas.csv', index=False)


