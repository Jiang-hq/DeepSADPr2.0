import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from numpy.core._multiarray_umath import ndarray, array
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from numpy import interp, math
import re
from collections import Counter

train_path = ""#训练集输入路径
indep_path = ""#独立测试集输入路径
acc_loss_savepath=""#loss曲线存储路径
model_savepath=""#模型存储路径
fw = open("" + str(j) + '.txt', 'w')  # 每一折的预测值存储路径
metrics_savepath="" #metrics矩阵存储路径

AA = 'ACDEFGHIKLMNPQRSTVWY'
def EAACFeature(sequence):
    patten = re.compile('X|U|_')
    windows = 5
    encodings = []
    seqLen = len(sequence[0])
    for i in sequence:
        code = []
        seqLine = list(i)
        for j in range(seqLen):
            if j < seqLen and j + windows <= seqLen:
                count = Counter(re.sub(patten, '', i[j:j + windows]))
                for key in count:
                    count[key] = count[key] / windows
                for aa in AA:
                    code.append(count[aa])
        encodings.append(code)
    return encodings


f1 = pd.read_csv(train_path)
f2 = pd.read_csv(indep_path)
seq2 = f2['sequence'].tolist()
x_train = f1['sequence'].tolist()
y_train = np.array(f1['label'].tolist())

# 训练集的序列、分类、训练集大小
x_ind = f2['sequence'].tolist()
y_ind = np.array(f2['label'].tolist())

train_encodings = np.array(EAACFeature(x_train))
test_encodings = np.array(EAACFeature(x_ind))
x_train = train_encodings.reshape(train_encodings.shape[0],37,20)
x_ind = test_encodings.reshape(test_encodings.shape[0],37,20)
shape = x_train.shape[1:]
shape1 = x_train.shape[1:]




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

    # 全连接层
    model.add(Dense(8, activation='relu'))  # dense代表全连接层，定义了8个节点
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


plt.figure(figsize=(20, 10))



matrix = []
kf = KFold(n_splits=10, random_state=30, shuffle=True)
j = 0
for train_index, test_index in kf.split(x_train):  # 将训练集进行10折交叉验证
    test_index1 = test_index.tolist()
    x_train3, x_test3 = x_train[train_index], x_train[test_index]
    y_train3, y_test3 = y_train[train_index], y_train[test_index]
    model = create_HYYY(shape=shape1)
    checkpoint = ModelCheckpoint(model_savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto',
                                 period=50,
                                 save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    callbacks_list = [early_stopping, checkpoint]

    hist = model.fit(x_train3, y_train3, validation_data=(x_test3, y_test3), epochs=500, batch_size=128,
                        shuffle=True, callbacks=callbacks_list, verbose=1)
    draw_loss_acc(hist)
    test_pred_proba = model.predict(x_test3) # 预测属于某一类的概率
    print(test_pred_proba)
    matrix.append(calculate_metrics(y_test3, test_pred_proba))
    draw_roc(y_test3, test_pred_proba)
    fw.write('score' + '\t' + 'label' + '\t' + 'position' + '\n')
    for t in range(0, len(test_pred_proba)):
        fw.write(str(test_pred_proba[t][0]))
        fw.write('\t')
        fw.write(str(y_test3[t]))
        fw.write('\t')
        fw.write(str(test_index1[t]))
        fw.write('\n')
    fw.close()


    if j == 9:
        print(auc_mean)
        print(print("CV AUC: %f" % mean(auc_mean)))

    j += 1

df = pd.DataFrame(matrix).to_csv("/media/deep-learning/jianghaoqiang/sha_project/model3/result_imbalance/CNN_EAAC/train_metrics.csv")
