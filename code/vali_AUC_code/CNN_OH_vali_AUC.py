import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from numpy.core._multiarray_umath import ndarray
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from numpy import interp, math

ROC_savepath=""#loss曲线存储路径
vali_resultpath=""#每一折的预测结果文件路径
metrics_savepath="" #metrics矩阵存储路径

def mean(a):
    return sum(a) / len(a)

def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):  # 计算阈值为0.5时的各性能指数
    my_metrics = {  # 先声明建立一个字典，对应KEY值
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
        if labels[i] == po_label:  # 如果为正样本
            if scores[i] >= cutoff:  # 阈值为0.5，如果打分大于0.5
                tp = tp + 1  # tp+1  预测为真，实际为真的
            else:
                fn = fn + 1  # 实际为真，预测为负
        else:  # 如果为负样本
            if scores[i] < cutoff:  # 打分小于阈值，说明实际为负，预测也为负
                tn = tn + 1  # tn+1
            else:
                fp = fp + 1  # 打分大于阈值，说明预测为正，实际为负

    my_metrics['SN'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'  # sn 灵敏度
    my_metrics['SP'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'  # sp 特异性
    my_metrics['ACC'] = (tp + tn) / (tp + fn + tn + fp)  # acc正确度
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ( tp + fp) * ( tp + fn) * ( tn + fp) * ( tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'  # 查准率
    my_metrics['Recall'] = my_metrics['SN']  # 召回率
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    return my_metrics


def draw_roc(true_data, predict ):
    fpr, tpr, thresholds = roc_curve(true_data, predict)  # roc_curve(y,scores) Y是标准值，score是阳性预测概率 thresholds每次取不同的阈值
    tprs.append(interp(mean_fpr, fpr, tpr))  # numpy.interp()主要使用场景为一维线性插值，返回离散数据的一维分段线性插值结果
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)     # 得出一组数据的AUC值
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.5f)' % (j, roc_auc))
    auc_mean.append(auc(fpr, tpr))

def draw_loss_acc(hist):
    # 画出loss图
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs = range(len(loss))
    plt.subplot(2, 5, j+1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')


    plt.plot(epochs, acc, label='Training acc')
    plt.plot(epochs, val_acc, label='Validation acc')
    plt.title('Training and validation acc and loss')  # 调整画板大小和字体大小

    plt.xlabel('Epoch')


    plt.legend()  # 绘制图例，默认在右上角
    plt.savefig(acc_loss_savepath)


tprs = []
auc_mean = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)  # 主要用于创建等差数列
plt.figure(figsize=(10, 10))

metrics_list = []
for j in range(0,10):

    '''导入分数以及验证集标签的地方，这里我为文件中分数的那一列设置标签为score'''
    f_vali = pd.read_csv(vali_result_path, sep='\t', header=0)
    y_test0 = f_vali['label']

    b = list(map(lambda x:float(x), y_test0))
    y_test0 = list(b)
    y_test = np.array(y_test0)  # 验证集的标签
    print('one_hot_result_' + str(j) + '.csv')
    score = f_vali['score'].tolist()

    c = list(map(lambda x:float(x), score))
    score = list(c)
    s = np.array(score)  # 预测的分数的标签
    print(s.shape)
    metrics = calculate_metrics(y_test, s)# 评估独立测试集
    draw_roc(y_test, s)
    metrics_list.append(metrics)
    if j == 9:
        print(auc_mean)
        print(print("indep AUC: %f" % mean(auc_mean)))

    j += 1


df = pd.DataFrame(metrics_list).to_csv(metrics_savepath)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),lw=2, alpha=.8)

plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.4f)' % (mean_auc),lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('Cross-Validation ROC of CNN',fontsize=18)
plt.legend(loc="lower right", prop={'size': 10})
plt.savefig(ROC_savepath)
plt.show()
