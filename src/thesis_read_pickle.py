from __future__ import print_function, division

__author__ = 'amrit'
import pickle
import os
import numpy as np

ROOT=os.getcwd()
MLS=["SVM",  "FFT1"]
#metrics=['accuracy','recall','precision','false_alarm']
metrics=['Dist2Heaven']
files=["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
features = ['10', '25', '50', '100']

def dump_files(f='',prefix=''):
    for _, _, files in os.walk(ROOT + "/../dump/thesis/"):
        for file in files:
            if file.startswith(prefix+f):
                return file

def for_LDADE():
    filenames=[]
    dic={}
    for f in files:
        filenames.append(dump_files(f,'LDADE_FFT_'))
    for f in filenames:
        if f!=None:
            with open("../dump/thesis/" + f, 'rb') as handle:
                g=f.split(".pickle")[0].split("LDADE_FFT_")[1]
                dic[g] = pickle.load(handle)
    return dic

def for_LDA():
    filenames=[]
    dic={}
    for f in files:
        filenames.append(dump_files(f,'LDA_FFT_'))

    for f in filenames:
        with open("../dump/thesis/" + f, 'rb') as handle:
            g = f.split(".pickle")[0].split("LDA_FFT_")[1]
            dic[g] = pickle.load(handle)
    return dic

if __name__ == '__main__':
    LDADE = for_LDADE()
    for i in files:
        if i!='pitsB':
            for j in MLS:
                print(i,j,np.median(LDADE[i][j]["Dist2Heaven"]))

    LDA = for_LDA()

    for i in files:
            for j in MLS:
                    print(i, j, np.median(LDA[i]['10'][j]["Dist2Heaven"]))