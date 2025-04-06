import numpy as np
import pickle

with open('../output_data3/cyclic_spectrum32.pickle', 'rb') as f:
    data_dict = pickle.load(f)


# 遍历字典中的每个键值对
for key, value_list in data_dict.items():
    # 遍历值列表中的每个 numpy 数组
    for i in range(len(value_list)):
        # 对 numpy 数组中的复数元素取幅值
        value_list[i] = np.abs(value_list[i])

# 打印处理后的字典（可选，用于验证结果）
for key, value_list in data_dict.items():
    print(f"Key: {key}, 第一个数组的形状: {value_list[0].shape}, 第一个数组的第一个元素: {value_list[0][0, 0]}")

with open('../output_data3/abs_cyclic_spectrum32.pickle','wb') as f:
    pickle.dump(data_dict,f)

with open('../output_data3/cyclic_spectrum16.pickle', 'rb') as f:
    data_dict1 = pickle.load(f)

# 遍历字典中的每个键值对
for key, value_list in data_dict1.items():
    # 遍历值列表中的每个 numpy 数组
    for i in range(len(value_list)):
        # 对 numpy 数组中的复数元素取幅值
        value_list[i] = np.abs(value_list[i])

# 打印处理后的字典（可选，用于验证结果）
for key, value_list1 in data_dict.items():
    print(f"Key: {key}, 第一个数组的形状: {value_list[0].shape}, 第一个数组的第一个元素: {value_list[0][0, 0]}")

with open('../output_data3/abs_cyclic_spectrum16.pickle', 'wb') as f:
    pickle.dump(data_dict1, f)

