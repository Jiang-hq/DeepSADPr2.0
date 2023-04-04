import pickle

import pandas as pd
from numpy import *
import numpy as np
import re
from collections import Counter
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import itertools  # 笛卡尔积
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 200)

AA = 'ACDEFGHIKLMNPQRSTVWY'

# (64384, 740)
def BLOSUM62(sequence):
    blosum62 = pd.read_csv(r'D:\sha-crosstalk-9-24\RF\BLOSUM62.csv')
    encodings = []
    for i in sequence:
        seqLine = list(i)
        code = []
        for seq in seqLine:
            code += list(blosum62[seq])
        encodings.append(code)
    return encodings


def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):
    my_metrics = {
        'SN': 'NA',
        'SP': 'NA',
        'ACC': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:
            if scores[i] >= cutoff:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if scores[i] < cutoff:
                tn = tn + 1
            else:
                fp = fp + 1

    my_metrics['SN'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'
    my_metrics['SP'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'
    my_metrics['ACC'] = (tp + tn) / (tp + fn + tn + fp)
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (
                                                                                                                     tp + fp) * (
                                                                                                                     tp + fn) * (
                                                                                                                     tn + fp) * (
                                                                                                                     tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
    my_metrics['Recall'] = my_metrics['SN']
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    return my_metrics


def calculate_metrics_list(data, label_column=0, score_column=2, cutoff=0.5, po_label=1):
    metrics_list = []
    for i in data:
        metrics_list.append(calculate_metrics(i[:, label_column], i[:, score_column], cutoff=cutoff, po_label=po_label))
    mean_dict = {}
    std_dict = {}
    keys = metrics_list[0].keys()
    for i in keys:
        mean_list = []
        for metric in metrics_list:
            mean_list.append(metric[i])
        mean_dict[i] = np.array(mean_list).sum() / len(metrics_list)
        std_dict[i] = np.array(mean_list).std()
    metrics_list.append(mean_dict)
    metrics_list.append(std_dict)
    return metrics_list


def save_prediction_metrics_list(metrics_list, output):
    with open(output, 'w') as f:
        f.write('Fold')
        for keys in metrics_list[0]:
            f.write('\t%s' % keys)
        f.write('\n')
        for i in range(len(metrics_list)):
            if i <= 9:
                f.write('%d' % (i + 1))
            elif i == 10:
                f.write('mean')
            else:
                f.write('std')
            for keys in metrics_list[i]:
                f.write('\t%s' % metrics_list[i][keys])
            f.write('\n')
    return None


def plot_roc_curve(data, output, label_column=0, score_column=2):
    tprs = []
    aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(data)):
        fpr, tpr, thresholds = roc_curve(data[i][:, label_column], data[i][:, score_column])
        fprArray.append(fpr)
        tprArray.append(tpr)
        thresholdsArray.append(thresholds)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    print(aucs)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink', 'cyan'])
    fig = plt.figure(0)
    for i, color in zip(range(len(fprArray)), colors):
        plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                 label='ROC fold %d (AUC = %0.4f)' % (i + 1, aucs[i]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    # mean()函数的功能是求取平均值，经常操作的参数是axis，以m*n的矩阵为例：axis = 0：压缩行，对各列求均值，返回1*n的矩阵
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # 计算标准差
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.9)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(output)
    plt.close(0)
    return mean_auc


def plot_prc_curve(data, output, label_column=0, score_column=2):
    precisions = []
    aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)

    for i in range(len(data)):
        precision, recall, _ = precision_recall_curve(data[i][:, label_column], data[i][:, score_column])
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        roc_auc = auc(recall, precision)
        aucs.append(roc_auc)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink', 'cyan'])
    # ROC plot for CV
    fig = plt.figure(0)
    for i, color in zip(range(len(recall_array)), colors):
        plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.7, color=color,label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, aucs[i]))
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)

    plt.plot(mean_recall, mean_precision, color='blue',
             label=r'Mean PRC (AUPRC = %0.4f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.9)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(output)
    plt.close(0)
    return mean_auc


def RF_Classifier(X, y, x_test, y_test, path,fold=10, n_trees=100):
    classes = sorted(list(set(y)))
    prediction_result_cv = []
    prediction_result_ind = []
    print('X.shape:',X.shape)
    print(y.shape)
    folds = StratifiedKFold(fold).split(X, y)

    for i, (trained, validated) in enumerate(folds):
        valid_index = validated.tolist()
        train_y, train_X = y[trained], X[trained]
        valid_y, valid_X = y[validated], X[validated]


        # model = LogisticRegression()
        model = RandomForestClassifier(max_depth=8, max_features='sqrt', min_samples_leaf=20,min_samples_split=300, n_estimators=n_trees)#,random_state=1


        rfc = model.fit(train_X, train_y)
        f = open(path + str(i) + '.pkl', 'wb')  # 保存模型
        pickle.dump(model, f)
        f.close()

        scores = rfc.predict_proba(valid_X)
        sco = rfc.predict_proba(valid_X)[:, 1]
        scores1 = rfc.predict_proba(x_test)
        sco1 = rfc.predict_proba(x_test)[:, 1]

        fw = open(path + 'RF_CV' + str(i) + '.txt', 'w')
        for t in range(0, len(scores)):
            fw.write(str(sco[t]))  # 预测值
            fw.write('\t')
            fw.write(str(valid_y[t]))  # 真实标签
            fw.write('\t')
            fw.write(str(valid_index[t]))  # 验证集的索引
            fw.write('\n')

        fw = open(path + 'RF_indep' + str(i) + '.txt', 'w')
        for t in range(0, len(scores1)):
            fw.write(str(sco1[t]))  # 预测值
            fw.write('\t')
            fw.write(str(y_test[t]))  # 真实标签
            fw.write('\t')
            # fw.write(str(y_test[t]))  # 验证集的索引
            fw.write('\n')

        tmp_result = np.zeros((len(valid_y), len(classes) + 1))
        tmp_result[:, 0], tmp_result[:, 1:] = valid_y, scores
        prediction_result_cv.append(tmp_result)

        tmp_result1 = np.zeros((len(y_test), len(classes) + 1))
        tmp_result1[:, 0], tmp_result1[:, 1:] = y_test, scores1
        prediction_result_ind.append(tmp_result1)
    header = 'n_trees: %d' % n_trees
    return header, prediction_result_cv, prediction_result_ind


def save_predict_result(data, output):
    with open(output, 'w') as f:
        for i in range(len(data)):
            f.write('# result for fold %d\n' % (i + 1))
            for j in range(len(data[i])):
                f.write('%d\t%s\n' % (data[i][j][0], data[i][j][2]))
    return None


def main():
    # 读取训练集、测试集数据
    raw_train = pd.read_csv("/media/deep-learning/jianghaoqiang/sha_project/model1/stacking/stacking_0.6_plan1/DB/CNN_based_vali_all",'\t')
    raw_test = pd.read_csv("/media/deep-learning/jianghaoqiang/sha_project/model1/CNN-based_trained_result/CD-HIT_0.6_plan1/data/INDEP.txt",',')
    # 训练集的序列、分类、训练集大小

    y = array(raw_train['label'])
    # 训练集的序列、分类、训练集大小
    y_ind = array(raw_test['label'])

    #train_encodings = np.array(BLOSUM62(X))
    #test_encodings = np.array(BLOSUM62(x_ind))
    train_encodings = pd.read_csv("/media/deep-learning/jianghaoqiang/sha_project/model1/stacking/stacking_0.6_plan1/DB/CNN_based_vali_five_model",'\t',header=None)
    test_encodings = pd.read_csv("/media/deep-learning/jianghaoqiang/sha_project/model1/stacking/stacking_0.6_plan1/DB/CNN_based_INDEP_five_model",'\t',header=None)

    new_train = np.array(train_encodings)
    #new_train = train_encodings.astype(np.float64)
    new_test = np.array(test_encodings)

    print(new_train.shape)
    print(new_test.shape)
    path1 = "/media/deep-learning/jianghaoqiang/sha_project/model1/stacking/stacking_0.6_plan1/result_RF_five_model_stacking/"
    para_info, cv_res, ind_res = RF_Classifier(new_train, y, new_test, y_ind, fold=10, n_trees=100,path=path1)

    return para_info, cv_res, ind_res, y


if __name__ == '__main__':
    trees_info, cv_result, ind_result, label = main()
    classify = sorted(list(set(label)))
    print(classify)
    out = "/media/deep-learning/jianghaoqiang/sha_project/model1/stacking/stacking_0.6_plan1/result_RF_five_model_stacking/"

    if len(classify) == 2:
        # 保存交叉验证结果
        save_predict_result(cv_result, out + '_pre_cv.txt')
        plot_roc_curve(cv_result, out + '_roc_cv.png', label_column=0, score_column=2)
        plot_prc_curve(cv_result, out + '_prc_cv.png', label_column=0, score_column=2)
        cv_metrics = calculate_metrics_list(cv_result, label_column=0, score_column=2, cutoff=0.5, po_label=1)
        save_prediction_metrics_list(cv_metrics, out + '_metrics_cv.txt')
        # 保存独立测试结果
        save_predict_result(ind_result, out + '_pre_ind.txt')
        plot_roc_curve(ind_result, out + '_roc_ind.png', label_column=0, score_column=2)
        plot_prc_curve(ind_result, out + '_prc_ind.png', label_column=0, score_column=2)
        ind_metrics = calculate_metrics_list(ind_result, label_column=0, score_column=2, cutoff=0.5, po_label=1)
        save_prediction_metrics_list(ind_metrics, out + '_metrics_ind.txt')
