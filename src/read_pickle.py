from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

ROOT=os.getcwd()
MLS=["DT", "RF", "SVM",  "FFT1"]
metrics=['accuracy','recall','precision','false_alarm']
files=["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
features = ['10', '25', '50', '100']

def dump_files(f='',prefix=''):
    for _, _, files in os.walk(ROOT + "/../dump/"):
        for file in files:
            if file.startswith(prefix+f):
                return file

def plot_LDADE_FFT(LDADE,LDA):
    tuned_med = {}
    tuned_iqr = {}
    for f in files:
        for m in metrics:
            if m not in tuned_med:
                tuned_med[m] = {}
                tuned_iqr[m] = {}
            if 'LDADE_FFT' not in tuned_med[m]:
                tuned_med[m]['LDADE_FFT'] = [round(np.median(LDADE[f]['FFT1'][m]), 2)]
                tuned_iqr[m]['LDADE_FFT'] = [
                    round(np.percentile(LDADE[f]['FFT1'][m], 75) - np.percentile(LDADE[f]['FFT1'][m], 25), 2)]
            else:
                tuned_med[m]['LDADE_FFT'].append(round(np.median(LDADE[f]['FFT1'][m]), 2))
                tuned_iqr[m]['LDADE_FFT'].append(
                    round(np.percentile(LDADE[f]['FFT1'][m], 75) - np.percentile(LDADE[f]['FFT1'][m], 25), 2))
            for x in features:
                if str(x) + '_FFT' not in tuned_med[m]:
                    tuned_med[m][str(x) + '_FFT'] = [round(np.median(LDA[f][x]['FFT1'][m]), 2)]
                    tuned_iqr[m][str(x) + '_FFT'] = [
                        round(np.percentile(LDA[f][x]['FFT1'][m], 75) - np.percentile(LDA[f][x]['FFT1'][m], 25), 2)]
                else:
                    tuned_med[m][str(x) + '_FFT'].append(round(np.median(LDA[f][x]['FFT1'][m]), 2))
                    tuned_iqr[m][str(x) + '_FFT'].append(
                        round(np.percentile(LDA[f][x]['FFT1'][m], 75) - np.percentile(LDA[f][x]['FFT1'][m], 25), 2))


    font = {'size': 80}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 8, 'legend.fontsize': 50, 'axes.labelsize': 100, 'legend.frameon': False,
             'figure.autolayout': True, 'axes.linewidth': 10}
    plt.rcParams.update(paras)
    fig, axes = plt.subplots(2,2,figsize=(100, 80))
    for i in range(2):
        for j in range(2):
            if i ==1:
                for k in tuned_med[metrics[i+j+1]]:
                    line, =axes[i, j].plot(xrange(6), tuned_med[metrics[i+j+1]][k], label=k+' med')
                    axes[i, j].plot(xrange(6), tuned_iqr[metrics[i + j+1]][k], linestyle="-.", color=line.get_color() )
                axes[i,j].set_xticks(range(6))
                axes[i, j].set_xticklabels(files)
                axes[i, j].set_ylabel(metrics[i+j+1], labelpad=30)
                axes[i, j].set_xlabel("Datasets", labelpad=30)
                axes[i, j].legend()
            else:
                for k in tuned_med[metrics[i + j]]:
                    line, = axes[i, j].plot(xrange(6), tuned_med[metrics[i + j]][k], label=k + ' med')
                    axes[i, j].plot(xrange(6), tuned_iqr[metrics[i + j]][k], linestyle="-.", color=line.get_color())
                axes[i,j].set_xticks(range(6))
                axes[i, j].set_xticklabels(files)
                axes[i, j].set_ylabel(metrics[i + j], labelpad=30)
                axes[i, j].set_xlabel("Datasets", labelpad=30)
                axes[i, j].legend()
    plt.savefig("../results/performance_LDA_FFT.png")
    plt.close(fig)


def plot_others_FFT(LDADE,LDA):
    tuned_med = {}
    tuned_iqr = {}
    for f in files:
        for m in metrics:
            if m not in tuned_med:
                tuned_med[m] = {}
                tuned_iqr[m] = {}
            if 'LDADE_FFT' not in tuned_med[m]:
                tuned_med[m]['LDADE_FFT'] = [round(np.median(LDADE[f]['FFT1'][m]), 2)]
                tuned_iqr[m]['LDADE_FFT'] = [
                    round(np.percentile(LDADE[f]['FFT1'][m], 75) - np.percentile(LDADE[f]['FFT1'][m], 25), 2)]
            else:
                tuned_med[m]['LDADE_FFT'].append(round(np.median(LDADE[f]['FFT1'][m]), 2))
                tuned_iqr[m]['LDADE_FFT'].append(
                    round(np.percentile(LDADE[f]['FFT1'][m], 75) - np.percentile(LDADE[f]['FFT1'][m], 25), 2))
            for x in features:
                if str(x) + '_FFT' not in tuned_med[m]:
                    tuned_med[m][str(x) + '_FFT'] = [round(np.median(LDA[f][x]['FFT1'][m]), 2)]
                    tuned_iqr[m][str(x) + '_FFT'] = [
                        round(np.percentile(LDA[f][x]['FFT1'][m], 75) - np.percentile(LDA[f][x]['FFT1'][m], 25), 2)]
                else:
                    tuned_med[m][str(x) + '_FFT'].append(round(np.median(LDA[f][x]['FFT1'][m]), 2))
                    tuned_iqr[m][str(x) + '_FFT'].append(
                        round(np.percentile(LDA[f][x]['FFT1'][m], 75) - np.percentile(LDA[f][x]['FFT1'][m], 25), 2))


    font = {'size': 80}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 8, 'legend.fontsize': 50, 'axes.labelsize': 100, 'legend.frameon': False,
             'figure.autolayout': True, 'axes.linewidth': 10}
    plt.rcParams.update(paras)
    fig, axes = plt.subplots(2,2,figsize=(100, 80))
    for i in range(2):
        for j in range(2):
            if i ==1:
                for k in tuned_med[metrics[i+j+1]]:
                    line, =axes[i, j].plot(xrange(6), tuned_med[metrics[i+j+1]][k], label=k+' med')
                    axes[i, j].plot(xrange(6), tuned_iqr[metrics[i + j+1]][k], linestyle="-.", color=line.get_color() )
                axes[i,j].set_xticks(range(6))
                axes[i, j].set_xticklabels(files)
                axes[i, j].set_ylabel(metrics[i+j+1], labelpad=30)
                axes[i, j].set_xlabel("Datasets", labelpad=30)
                axes[i, j].legend()
            else:
                for k in tuned_med[metrics[i + j]]:
                    line, = axes[i, j].plot(xrange(6), tuned_med[metrics[i + j]][k], label=k + ' med')
                    axes[i, j].plot(xrange(6), tuned_iqr[metrics[i + j]][k], linestyle="-.", color=line.get_color())
                axes[i,j].set_xticks(range(6))
                axes[i, j].set_xticklabels(files)
                axes[i, j].set_ylabel(metrics[i + j], labelpad=30)
                axes[i, j].set_xlabel("Datasets", labelpad=30)
                axes[i, j].legend()
    plt.savefig("../results/performance_LDA_FFT.png")
    plt.close(fig)

def for_LDADE():
    filenames=[]
    dic={}
    for f in files:
        filenames.append(dump_files(f,'LDADE'))
    for f in filenames:
        with open("../dump/" + f, 'rb') as handle:
            g=f.split(".pickle")[0].split("LDADE")[1]
            dic[g] = pickle.load(handle)
    return dic

def for_LDA():
    filenames=[]
    dic={}
    for f in files:
        filenames.append(dump_files(f,'LDA'))
    for f in filenames:
        with open("../dump/" + f, 'rb') as handle:
            g = f.split(".pickle")[0].split("LDA")[1]
            dic[g] = pickle.load(handle)
    return dic


if __name__ == '__main__':
    LDADE=for_LDADE()
    LDA=for_LDA()
    plot_LDADE_FFT(LDADE,LDA)
