import re
from typing import List, Any, Union

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
import  random


train_path = ""#训练集输入路径
indep_path = ""#独立测试集输入路径
acc_loss_savepath=""#loss曲线存储路径
model_savepath=""#模型存储路径
fw = open("" + str(j) + '.txt', 'w')  # 每一折的预测值存储路径
metrics_savepath="" #metrics矩阵存储路径

def read_svm(file):
    encodings = []
    labels = []
    with open(file) as f:
        records = f.readlines()
        random.shuffle(records)
    for line in records:
        line = re.sub('\d+:', '', line)
        array = line.strip().split() if line.strip() != '' else None
        encodings.append(array[1:])
        labels.append(int(array[0]))

    return np.array(encodings).astype(float), np.array(labels).astype(int)

AA = 'GAVLIFWYDNEKQMSTCPHR_'
def pep1(path):
    seqs = open(path).readlines()
    cut = 0
    X = [[AA.index(res.upper()) if res.upper() in AA else 0
          for res in (seq.split(',')[0][cut:-cut] if cut != 0 else seq.split(',')[0])]
         for seq in seqs if seq.strip() != '']
    y = [int(seq.split(',')[1]) for seq in seqs if seq.strip() != '']

    return np.array(X), np.array(y)

def mean(a):
    return sum(a) / len(a)

def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):   # 计算阈值为0.5时的各性能指数
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
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (
        tp + fn) * (tn + fp) * (tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
    my_metrics['Recall'] = my_metrics['SN']
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    return my_metrics

def draw_roc(true_data,predict):
    fpr, tpr, thresholds = roc_curve(true_data, predict)  # roc_curve(y,scores) Y是标准值，score是阳性预测概率
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.3f)' % (j, roc_auc))
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
    plt.title('Training and validation accuracy')  # 调整画板大小和字体大小

    plt.xlabel('Epoch')


    plt.legend()  # 绘制图例，默认在右上角
    plt.savefig(acc_loss_savepath)

def create_HXXX(input_length=41,dropout=0.5):
    model = Sequential() # 定义顺序模型
    model.add(Embedding(21, 32, input_length=input_length))
    model.add(Conv1D(128, 1, activation='relu')) # 通过.add()方法一个个的将layer加入模型中
    model.add(Dropout(0.7))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Conv1D(128, 9, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.7))
    model.add(Conv1D(128, 10, activation='relu',padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.7))

    # 全连接层
    model.add(Dense(8,activation='relu')) # dense代表全连接层，定义了8个节点
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

x_train, y_train = pep1(path=train_path)
x_test, y_test = pep1(path=indep_path)


auc_mean=[]
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)  # 主要用于创建等差数列
plt.figure(figsize=(10,10))



plt.figure(figsize=(20,10))


matrix = []
kf = KFold(n_splits = 10,random_state=30,shuffle=True)
j = 0
for train_index, test_index in kf.split(x_train):
    # 将训练集进行10折交叉验证
    test_index1 = test_index.tolist()
    x_train3, x_test3 = x_train[train_index], x_train[test_index]
    y_train3, y_test3 = y_train[train_index], y_train[test_index]

    model = create_HXXX(input_length=41)
    checkpoint = ModelCheckpoint(model_savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=50,save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    callbacks_list = [early_stopping, checkpoint]
    hist = model.fit(x_train3, y_train3, validation_data=(x_test3, y_test3), epochs=500, batch_size=64,shuffle=True, callbacks=callbacks_list, verbose=1)
    draw_loss_acc(hist)
    test_pred_proba = model.predict(x_test3)  # 预测属于某一类的概率
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

df = pd.DataFrame(matrix).to_csv(metrics_savepath)  # 生成性能指标