from __future__ import print_function, division

__author__ = 'amrit'

import sys

#sys.dont_write_bytecode = True
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.text as mpl_text
import numpy as np

ROOT=os.getcwd()
MLS=["DT", "RF", "SVM",  "FFT1"]
#metrics=['accuracy','recall','precision','false_alarm']
metrics=['recall','precision']
files=["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
features = ['10', '25', '50', '100']

class AnyObject(object):
    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline',
                                horizontalalignment=u'left', multialignment=None,
                                fontproperties=None, linespacing=None,
                                rotation_mode=None)
        handlebox.add_artist(patch)
        return patch

def dump_files(f='',prefix=''):
    for _, _, files in os.walk(ROOT + "/../dump/"):
        for file in files:
            if file.startswith(prefix+f):
                return file

def LDADE_FFT(LDADE,LDA):
    tuned_med = {}
    for f in files:
        for m in metrics:
            if m not in tuned_med:
                tuned_med[m] = {}
            if f not in tuned_med[m]:
                tuned_med[m][f]=[]
                tuned_med[m][f].append(['LDADE_FFT'] + LDADE[f]['FFT1'][m])
                tuned_med[m][f].append(['LDADE_SVM'] + LDADE[f]['SVM'][m])

            for x in features:
                    tuned_med[m][f].append([str(x) + '_FFT'] + LDA[f][x]['FFT1'][m])
    return tuned_med

def FFT_features(LDA):
    tuned_med = {}
    for f in files:
        for m in ['features']:
            if m not in tuned_med:
                tuned_med[m] = {}
            if f not in tuned_med[m]:
                tuned_med[m][f]=[]
            for x in ['10']:
                    tuned_med[m][f].append([str(x) + '_FFT'] + LDA[f][x]['FFT1'][m])
    return tuned_med

def LDADE_FFT_runtimes(LDADE):
    tuned_med = {}
    for f in files:
        for m in ['times']:
            if m not in tuned_med:
                tuned_med[m] = {}
            if f not in tuned_med[m]:
                tuned_med[m][f]=[]
                tuned_med[m][f].append(['LDADE_FFT'] + [np.median(LDADE[f]['FFT1'][m])])
                tuned_med[m][f].append(['LDADE_SVM'] + [np.median(LDADE[f]['SVM'][m])])

    return tuned_med

def SVM_FFT(LDA,untuned):
    tuned_med = {}
    for f in files:
        for m in metrics:
            if m not in tuned_med:
                tuned_med[m] = {}
            if f not in tuned_med[m]:
                tuned_med[m][f]=[]
                tuned_med[m][f].append(['TFIDF_SVM'] + untuned[f]['TFIDF']['SVM'][m])

            for x in features:
                    tuned_med[m][f].append([str(x) + '_FFT'] + LDA[f][x]['FFT1'][m])
    return tuned_med

def SVM_FFT_runtimes(LDA,untuned):
    tuned_med = {}
    for f in files:
        for m in ['times']:
            if m not in tuned_med:
                tuned_med[m] = {}
            if f not in tuned_med[m]:
                tuned_med[m][f]=[]
                tuned_med[m][f].append(['TFIDF_SVM'] + [np.median(untuned[f]['TFIDF']['SVM'][m])])

            for x in features:
                    tuned_med[m][f].append([str(x) + '_FFT'] + [np.median(LDA[f][x]['FFT1'][m])])
    return tuned_med

def draw(dic):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 70, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
                  'figure.autolayout': True,'axes.linewidth':8}
    plt.rcParams.update(paras)

    boxprops = dict(linewidth=9,color='black')
    colors=['red','green', 'blue', 'orange','cyan','purple']*6
    whiskerprops = dict(linewidth=5)
    medianprops = dict(linewidth=8, color='firebrick')
    #meanpointprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick',markersize=20)

    fig = plt.figure(figsize=(80, 60))
    outer = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.2)
    for i,a in enumerate([1]):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.05, hspace=0.0)
        for j,b in enumerate(dic.keys()):
            ax = plt.Subplot(fig, inner[j])
            temp=[item[1:] for sublist in dic[b].values() for item in sublist]

            bplot=ax.boxplot(temp,showmeans=False,showfliers=False,medianprops=medianprops,capprops=whiskerprops,
                       flierprops=whiskerprops,boxprops=boxprops,whiskerprops=whiskerprops,
                       positions=[1,2,3,4,5,6, 8,9,10,11,12,13 ,15,16,17,18,19,20 ,22,23,24,25,26,27 ,29,30,31,32,33,34,
                                  36,37,38,39,40,41])
            for patch, color in zip(bplot['boxes'], colors):
                patch.set(color=color)
            ax.set_xticks([3.5,11.5,17.5,24.5,31.5,38.5])
            ax.set_xticklabels(dic[b].keys())
            ax.set_ylabel(b,labelpad=30)
            #ax.set_ylim([0,1])
            if j!=1:
                plt.setp(ax.get_xticklabels(), visible=False)
            fig.add_subplot(ax)

    # box1 = TextArea("DT", textprops=dict(color=colors[0],size='large'))
    # box2 = TextArea("RF", textprops=dict(color=colors[1],size='large'))
    # box3 = TextArea("SVM", textprops=dict(color=colors[2],size='large'))
    # box = HPacker(children=[box1, box2, box3],
    #               align="center",
    #               pad=0, sep=5)
    #
    # anchored_box = AnchoredOffsetbox(loc=3,child=box, pad=0.,frameon=True,
    #                                  bbox_to_anchor=(0., 1.02),borderpad=0.)
    #
    # plt.artist(anchored_box)
    obj_0 = AnyObject("LDADE_FFT", colors[0])
    obj_1 = AnyObject("LDADE_SVM", colors[1])
    obj_2 = AnyObject("10_FFT", colors[2])
    obj_3 = AnyObject("25_FFT", colors[3])
    obj_4 = AnyObject("50_FFT", colors[4])
    obj_5 = AnyObject("100_FFT", colors[5])

    plt.legend([obj_0, obj_1,obj_2,obj_3,obj_4,obj_5], ['LDADE_FFT','LDADE_SVM', '10_FFT', '25_FFT','50_FFT','100_FFT'],
               handler_map={obj_0: AnyObjectHandler(), obj_1: AnyObjectHandler(),obj_2: AnyObjectHandler(),
               obj_3: AnyObjectHandler(), obj_4: AnyObjectHandler(),obj_5: AnyObjectHandler()},
               loc='upper center', bbox_to_anchor=(0.5, 2.1),
               fancybox=True, shadow=True, ncol=6,handletextpad=4)
    # plt.figtext(0.40, 0.9, 'DT', color=colors[0],size='large')
    # plt.figtext(0.50, 0.9, 'RF', color=colors[1],size='large')
    # plt.figtext(0.60, 0.9, 'SVM', color=colors[2],size='large')

    plt.savefig("../results/graph1.png", bbox_inches='tight')
    plt.close(fig)

def draw1(dic):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 70, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
                  'figure.autolayout': True,'axes.linewidth':8}
    plt.rcParams.update(paras)

    boxprops = dict(linewidth=9,color='black')
    colors=['red','green', 'blue', 'orange','purple']*6
    whiskerprops = dict(linewidth=5)
    medianprops = dict(linewidth=8, color='firebrick')
    #meanpointprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick',markersize=20)

    fig = plt.figure(figsize=(80, 60))
    outer = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.2)
    for i,a in enumerate([1]):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.05, hspace=0.0)
        for j,b in enumerate(dic.keys()):
            ax = plt.Subplot(fig, inner[j])
            temp=[item[1:] for sublist in dic[b].values() for item in sublist]

            bplot=ax.boxplot(temp,showmeans=False,showfliers=False,medianprops=medianprops,capprops=whiskerprops,
                       flierprops=whiskerprops,boxprops=boxprops,whiskerprops=whiskerprops,
                       positions=[1,2,3,4,5, 7,8,9,10,11, 13,14,15,16,17 ,19,20,21,22,23 ,25,26,27,28,29, 31,32,33,34,35
                                  ])
            for patch, color in zip(bplot['boxes'], colors):
                patch.set(color=color)
            ax.set_xticks([3,9,15,21,27,33])
            ax.set_xticklabels(dic[b].keys())
            ax.set_ylabel(b,labelpad=30)
            #ax.set_ylim([0,1])
            if j!=1:
                plt.setp(ax.get_xticklabels(), visible=False)
            fig.add_subplot(ax)

    # box1 = TextArea("DT", textprops=dict(color=colors[0],size='large'))
    # box2 = TextArea("RF", textprops=dict(color=colors[1],size='large'))
    # box3 = TextArea("SVM", textprops=dict(color=colors[2],size='large'))
    # box = HPacker(children=[box1, box2, box3],
    #               align="center",
    #               pad=0, sep=5)
    #
    # anchored_box = AnchoredOffsetbox(loc=3,child=box, pad=0.,frameon=True,
    #                                  bbox_to_anchor=(0., 1.02),borderpad=0.)
    #
    # plt.artist(anchored_box)
    obj_0 = AnyObject("TFIDF_SVM", colors[0])
    obj_1 = AnyObject("10_FFT", colors[1])
    obj_2 = AnyObject("25_FFT", colors[2])
    obj_3 = AnyObject("50_FFT", colors[3])
    obj_4 = AnyObject("100_FFT", colors[4])

    plt.legend([obj_0, obj_1,obj_2,obj_3,obj_4], ['TFIDF_SVM', '10_FFT', '25_FFT','50_FFT','100_FFT'],
               handler_map={obj_0: AnyObjectHandler(), obj_1: AnyObjectHandler(),obj_2: AnyObjectHandler(),
               obj_3: AnyObjectHandler(), obj_4: AnyObjectHandler()},
               loc='upper center', bbox_to_anchor=(0.5, 2.1),
               fancybox=True, shadow=True, ncol=5,handletextpad=4)
    # plt.figtext(0.40, 0.9, 'DT', color=colors[0],size='large')
    # plt.figtext(0.50, 0.9, 'RF', color=colors[1],size='large')
    # plt.figtext(0.60, 0.9, 'SVM', color=colors[2],size='large')

    plt.savefig("../results/graph2.png", bbox_inches='tight')
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

def for_untuned():
    filenames=[]
    dic={}
    for f in files:
        filenames.append(dump_files(f,'untuned'))
    for f in filenames:
        with open("../dump/" + f, 'rb') as handle:
            g = f.split(".pickle")[0].split("untuned")[1]
            dic[g] = pickle.load(handle)
    return dic


if __name__ == '__main__':
    LDADE=for_LDADE()
    LDA=for_LDA()
    untuned = for_untuned()
    dic=LDADE_FFT(LDADE,LDA)
    ## RQ2
    print(dic)
    #draw(dic)

    dic=SVM_FFT(LDA,untuned)
    ## RQ1
    print(dic)
    #draw1(dic)

    # dic=SVM_FFT_runtimes(LDA,untuned)
    # print(dic)
    # dic=LDADE_FFT_runtimes(LDADE)
    # print(dic)

    # print(FFT_features(LDA))

