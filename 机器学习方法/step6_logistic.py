import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def create_feture_sel_model(X, Y):
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X, Y)
    return model

def get_feas_impoortance(X, Y, top_k):
    # X,Y 都是dataframe结构
    #print("select feature...")
    select_feature_model = create_feture_sel_model(X, Y)
    importances = select_feature_model.feature_importances_

    cols = X.columns
    pd1 = pd.DataFrame(importances, cols, columns=['importance'])
    pd1.sort_values(by='importance', ascending=True).plot(kind='barh', figsize=(4, 10))
    plt.grid()
    #plt.show()
    # 用置信区间来计算相对重要性的估计值

    std = np.std([i.feature_importances_ for i in select_feature_model.estimators_], axis=0)
    feat_with_importance = pd.Series(importances, X.columns)
    fig, ax = plt.subplots(figsize=(12, 5))
    feat_with_importance.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decraese in impurity")
    plt.grid()
    #plt.show()
    #print("feature selecting finish")
    # 按重要性的降序来选着前K个特征
    X_result = X.iloc[:, select_feature_model.feature_importances_.argsort()[::-1][:top_k]]
    #print(f'select {top_k} features')
    select_feas=X_result.columns.tolist()
    return select_feas



if __name__ == '__main__':
    data_df = pd.read_csv('../output_data1/nor_all_feas.csv', encoding='UTF-8')
    SNR_list = list(i for i in range(-20, 19, 2))
    metrics_df = pd.DataFrame(
        columns=['SNR', 'train_acc', 'test_acc', 'roc_auc_score', 'f1_score', 'precision_score', 'recall_score'])

    for select_SNR in SNR_list:
        print("信噪比:", select_SNR)
        sampled_df = data_df[data_df['SNR'] == select_SNR].reset_index(drop=True)
        # print(sampled_df.info())
        X_df = sampled_df.drop(columns=['index', 'Modulation', 'SNR']).reset_index(drop=True)
        # print(X_df.info())
        Y_df = sampled_df['Modulation'].reset_index(drop=True)
        # print(Y_df.info())
        """mapping_dict = {
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
        sampled_df['Modulation']=sampled_df['Modulation'].map(mapping_dict)
        Y_df=sampled_df['Modulation']"""
        top_k = 12
        selected_features = get_feas_impoortance(X_df, Y_df, top_k)
        print("选择的特征：")
        print(selected_features)

        # print(selected_features)
        # 选择训练模型的特征
        X_data = X_df[selected_features]
        # print(data_df.info())
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_df, test_size=0.3, random_state=42)

        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
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

        # 将当前信噪比下的指标添加到 DataFrame 中
        metrics_df = metrics_df.append({
            'SNR': select_SNR,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'roc_auc_score': auc_value,
            'f1_score': f1,
            'precision_score': precision,
            'recall_score': recall
        }, ignore_index=True)

    #metrics_df.to_csv('../output_data3/Logistic.csv', index=False)







