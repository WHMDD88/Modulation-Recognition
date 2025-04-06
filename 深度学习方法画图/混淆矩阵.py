import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, \
    recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def create_feture_sel_model(X, Y):
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X, Y)
    return model

def get_feas_impoortance(X, Y, top_k):
    # X,Y 都是dataframe结构
    # print("select feature...")
    select_feature_model = create_feture_sel_model(X, Y)
    # 按重要性的降序来选着前K个特征
    X_result = X.iloc[:, select_feature_model.feature_importances_.argsort()[::-1][:top_k]]
    # print(f'select {top_k} features')
    select_feas = X_result.columns.tolist()
    return select_feas

if __name__ == '__main__':
    data_df = pd.read_csv('../output_data1/nor_all_feas.csv', encoding='UTF-8')
    SNR_list = [16,18]
    metrics_df = pd.DataFrame(
        columns=['SNR', 'train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score'])

    # 定义标签映射字典
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
        'QAM64': 10,
    }

    for select_SNR in SNR_list:
        print("信噪比:", select_SNR)
        sampled_df = data_df[data_df['SNR'] == select_SNR]

        X_df = sampled_df.drop(columns=['index', 'Modulation', 'SNR']).reset_index(drop=True)
        Y_df = sampled_df[['Modulation']].reset_index(drop=True)

        # 将标签转换为数值类型
        # Y_df['Modulation'] = Y_df['Modulation'].map(mapping_dict)

        top_k = 12
        selected_features = get_feas_impoortance(X_df, Y_df['Modulation'], top_k)
        print("选择的特征：")
        print(selected_features)

        X_data = X_df[selected_features]

        X_Y = X_data.copy()
        X_Y['Modulation'] = Y_df['Modulation']

        X_train, X_valid, y_train, y_valid = train_test_split(X_data, Y_df['Modulation'], test_size=0.11,
                                                              random_state=42)

        # 按标签比例抽样
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.19, random_state=42)
        for _, test_index in sss.split(X_Y, X_Y['Modulation']):
            XY_df = X_Y.iloc[test_index]
        X1_df = XY_df.drop(columns=['Modulation'])
        Y1_df = XY_df['Modulation']

        X_test = pd.concat([X_valid, X1_df], axis=0, ignore_index=True)
        y_test = pd.concat([y_valid, Y1_df], axis=0, ignore_index=True)
        # 模型
        clf = lgb.LGBMClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test.values)
        y_hat = clf.predict(X_train.values)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print('**********************************************************************')
        print()
        test_acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_hat)
        print("test_acc is {:.5}".format(test_acc))
        print("train_acc is {:.5}".format(train_acc))
        print(classification_report(y_test, y_pred, digits=3))
        auc_value = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
        print('roc_auc_score is {:.5}'.format(auc_value))
        f1 = f1_score(y_test, y_pred, average='micro')
        print('f1_score is {:.5}'.format(f1))
        precision = precision_score(y_test, y_pred, average='micro')
        print('precision score is {:.5}'.format(precision))
        recall = recall_score(y_test, y_pred, average='micro')
        print('recall score is {:.5}'.format(recall))
        print("********************************************************")
        print()

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 绘制混淆矩阵
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap=plt.cm.Greens, values_format='d', colorbar=False)

        # 新增标签调整代码
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(rotation=45, ha='right', fontsize=8,family='Time News Roman')
        plt.yticks(fontsize=8,family='Time News Roman')
        #plt.title(f'Confusion Matrix at SNR = {select_SNR}')
        plt.tight_layout()
        #plt.savefig('CM/' + str(select_SNR) + 'CM.svg')
        plt.show()
        metrics_df = metrics_df.append({
            'SNR': select_SNR,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'roc_auc_score': auc_value,
            'f1_score': f1,
            'precision_score': precision,
            'recall_score': recall
        }, ignore_index=True)

