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


train_path = ""#训练集输入路径
indep_path = ""#独立测试集输入路径
ROC_savepath=""#loss曲线存储路径
model_savepath=""#模型存储路径
fw = open("" + str(j) + '.txt', 'w')  # 每一折的预测值存储路径
metrics_savepath="" #metrics矩阵存储路径


letterDict = {}
letterDict["A"] = 0
letterDict["C"] = 1
letterDict["D"] = 2
letterDict["E"] = 3
letterDict["F"] = 4
letterDict["G"] = 5
letterDict["H"] = 6
letterDict["I"] = 7
letterDict["K"] = 8
letterDict["L"] = 9
letterDict["M"] = 10
letterDict["N"] = 11
letterDict["P"] = 12
letterDict["Q"] = 13
letterDict["R"] = 14
letterDict["S"] = 15
letterDict["T"] = 16
letterDict["V"] = 17
letterDict["W"] = 18
letterDict["Y"] = 19
letterDict["_"] = 20

depth = len(letterDict)
print(depth)

f2 = pd.read_csv(test_path)
seq2 = f2['sequence'].tolist()
all = []
all2 = []

for each in seq2:
    indics2 = []
    for i in each:
        if i in letterDict:
            indics2.append(letterDict[i])
        else:
            print(i)
            indics2.append(letterDict["_"])
    all2.append(indics2)

depth = len(letterDict)

# 方法一：选择tensor包调用

x_train0 = tf.one_hot(indices=all, depth=depth)
x_train = x_train0.numpy()

shape1 = x_train.shape[1:]

x_test0 = tf.one_hot(indices=all2, depth=depth)
x_test = np.asarray(x_test0)  # 验证集的序列

y_test0 = f2['label'].tolist()
y_test = np.array(y_test0)    # 验证集的标签

shape2 = x_test.shape[1:]
print(shape2)


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


def draw_roc(true_data, predict):
    fpr, tpr, thresholds = roc_curve(true_data, predict)  # roc_curve(y,scores) Y是标准值，score是阳性预测概率 thresholds每次取不同的阈值
    tprs.append(interp(mean_fpr, fpr, tpr))  # numpy.interp()主要使用场景为一维线性插值，返回离散数据的一维分段线性插值结果
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)     # 得出一组数据的AUC值
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.5f)' % (j, roc_auc))
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
    plt.savefig("/media/deep-learning/jianghaoqiang/sha_project/model2/result_balance/CNN_OH/INDEPacc-loss.jpg")


def create_HYYY(shape):
    model = Sequential()  # 定义顺序模型
    model.add(Conv1D(128, 1, activation='relu', input_shape=shape))  # 通过.add()方法一个个的将layer加入模型中
    model.add(Dropout(0.7))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Conv1D(128, 9, activation='relu'))
    model.add(MaxPooling1D(2))    # 池化 进行下采样 大小为2，步长为1
    model.add(Dropout(0.7))
    model.add(Conv1D(128, 10, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.7))


    model.add(Dense(8, activation='relu'))  # dense代表全连接层，定义了8个节点
    model.add(Dropout(0.3))
    model.add(GlobalAveragePooling1D())

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


tprs = []
auc_mean = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)  # 主要用于创建等差数列
plt.figure(figsize=(10, 10))

# 测试阶段，加载10折交叉的每个模型，对独立测试集进行测试
model = create_HYYY(shape=shape2)

list = []


for j in range(0,10):
    model.load_weights(model_savepath)
    s = model.predict_proba(x_test)
    print(s.shape)
    metrics = calculate_metrics(y_test, s)# 评估独立测试集
    for t in range(0, len(s)):
        fw.write(str(s[t][0]))
        fw.write('\t')
        fw.write(str(y_test[t]))
        fw.write('\n')
    fw.close()
    draw_roc(y_test, s)
    list.append(metrics)
    if j == 9:
        print(auc_mean)
        print(print("indep AUC: %f" % mean(auc_mean)))

    j += 1


df = pd.DataFrame(list).to_csv(metrics_savepath)
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
