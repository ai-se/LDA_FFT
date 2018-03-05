from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
import numpy as np
import pandas as pd
from FFT import FFT

metrics=['accuracy','recall','precision','false_alarm']

metrics_dic={'accuracy':-2,'recall':-6,'precision':-7,'false_alarm':-4}

def DT(k,train_data,train_labels,test_data,test_labels, metric):

    model = DecisionTreeClassifier(**k)
    model.fit(train_data, train_labels)
    prediction=model.predict(test_data)
    dic = {}

    for i in metrics:
        dic[i] = round(evaluation(i, prediction, test_labels),3)

    return dic[metric], [dic, model.feature_importances_]

def RF(k,train_data,train_labels,test_data,test_labels, metric):
    model = RandomForestClassifier(**k)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    dic = {}
    for i in metrics:
        dic[i] = round(evaluation(i, prediction, test_labels),3)
    return dic[metric], [dic, model.feature_importances_]


def SVM(k,train_data,train_labels,test_data,test_labels, metric):
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    train_data = scaling.transform(train_data)
    test_data = scaling.transform(test_data)
    model = SVC(cache_size=20000,**k)
    model.fit(train_data, train_labels)
    #print(model.coef_)
    prediction = model.predict(test_data)
    dic = {}
    for i in metrics:
        dic[i] = round(evaluation(i, prediction, test_labels),3)
    return dic[metric], [dic, []]


def FFT1(k,train_data,train_labels,test_data,test_labels, metric):
    dic={}
    for i in metrics:
        fft = FFT(max_level=5)
        fft.criteria=i
        train_labels=np.reshape(train_labels,(-1,1))
        test_labels = np.reshape(test_labels, (-1, 1))

        training=np.hstack((train_data, train_labels))
        testing = np.hstack((test_data, test_labels))
        training_df = pd.DataFrame(training)
        testing_df = pd.DataFrame(testing)
        training_df.rename(columns={training_df.columns[-1]: "bug"},inplace=True)
        testing_df.rename(columns={testing_df.columns[-1]: "bug"},inplace=True)

        fft.target = "bug"
        fft.train, fft.test = training_df, testing_df
        fft.build_trees()  # build and get performance on TEST data
        t_id = fft.find_best_tree()  # find the best tree on TRAIN data
        fft.eval_tree(t_id)  # eval all the trees on TEST data

        dic[i]=fft.performance_on_test[t_id][metrics_dic[i]]
    return dic[metric], [dic,[]]


def evaluation(measure, prediction, test_labels, class_target=1):
    confu = confusion_matrix(test_labels, prediction)
    fp = confu.sum(axis=0) - np.diag(confu)
    fn = confu.sum(axis=1) - np.diag(confu)
    tp = np.diag(confu)
    tn = confu.sum() - (fp + fn + tp)
    if measure == "accuracy":
        return accuracy_score(test_labels, prediction)
    if measure == "recall":
        recall = 0
        if class_target == -1:
            for m in range(len(tp)):
                if tp[m] != 0 and (tp[m] + fn[m]) != 0:
                    recall += float(tp[m]) / (tp[m] + fn[m])
            return recall / len(tp)
        else:
            if tp[class_target] != 0 and (tp[class_target] + fn[class_target]) != 0:
                return float(tp[class_target]) / (tp[class_target] + fn[class_target])
            else:
                return 0.0

    if measure == "precision":
        precision = 0
        if class_target == -1:
            for m in range(len(tp)):
                if tp[m] != 0 and (tp[m] + fp[m]) != 0:
                    precision += float(tp[m]) / (tp[m] + fp[m])
            return precision / len(tp)
        else:
            if tp[class_target] != 0 and (tp[class_target] + fp[class_target]) != 0:
                return float(tp[class_target]) / (tp[class_target] + fp[class_target])
            else:
                return 0.0
    if measure == "false_alarm":
        fals=0
        if class_target==-1:
            for m in range(len(fp)):
                if fp[m] != 0 and (fp[m] + tn[m]) !=0 :
                    fals+=float(fp[m])/(fp[m]+tn[m])
            return fals/len(fp)
        else:
            if fp[class_target] != 0 and (fp[class_target] + tn[class_target]) != 0:
                return float(fp[class_target]) / (fp[class_target] + tn[class_target])
            else:
                return 0.0

    if measure == "f1":
        if class_target==-1:
            return f1_score(test_labels, prediction, average='macro')
        else:
            return f1_score(test_labels, prediction, pos_label=class_target, average='binary')