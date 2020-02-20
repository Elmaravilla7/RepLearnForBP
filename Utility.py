from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score, adjusted_rand_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from operator import itemgetter
from sklearn.tree import DecisionTreeClassifier

import loadXES

def NMIscore(path1,path2):
    df = np.asarray(pd.read_csv(path2))
    dt=np.asarray(pd.read_csv(path1))
    x = []
    t = []
    for i in df:
        x.append(i[1])
    for j in dt:
        t.append(j[1])
    if (len(t) > len(x)):
         t= t[:len(x)]
    if(len(x)>len(t)):
        x=x[:len(t)]
    return str(normalized_mutual_info_score(t, x,average_method='arithmetic'))

def Rscore(path1,path2):
    df = np.asarray(pd.read_csv(path2))
    dt=np.asarray(pd.read_csv(path1))
    x = []
    t = []
    for i in df:
        x.append(i[1])
    for j in dt:
        t.append(j[1])
    if (len(t) > len(x)):
         t= t[:len(x)]
    if (len(x) > len(t)):
        x = x[:len(t)]
    return str(adjusted_rand_score(t, x))

def AUC(path1,path2,lw=None):
    df = np.asarray(pd.read_csv(path2))
    dt = np.asarray(pd.read_csv(path1))
    X = {}
    y ={}
    for i in df:
        y[i[0]]=i[1]
    for j in dt:
        X[j[0]]=j[1]
    tmp1=[]
    tmp2=[]
    for key in X:
        tmp1.append(X[key])
    for key in y:
        tmp2.append(y[key])
    X=tmp1
    y=tmp2
    if(len(X)>len(y)):
        X=X[:len(y)]
    if (len(y) > len(X)):
        y = y[:len(X)]
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    X_train=np.asarray(X_train).reshape(-1,1)
    X_test=np.asarray(X_test).reshape(-1,1)
    classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()
    return str(roc_auc_score(y_test,y_score))



def dizionario(v):
    diz={}
    for element in v:
        for i in range (0,len(element)):
            if(element[i]!='ArtificialStartTask' and element[i]!='ArtificialEndTask'):
                if (element[i] in diz):
                    diz[element[i]]=diz[element[i]]+1
                else:
                    diz[element[i]]=1
    return diz

def dizionarioX(v):
    diz = {}
    for element in v:
        for i in range (0,len(element)):
            if(element[i]!=[]):
                if(element[i][0]!='artificial'):
                    if (element[i][0] in diz):
                        diz[element[i][0]]=diz[element[i][0]]+1
                    else:
                        diz[element[i][0]]=1
    return diz

def freqNumerica(logName):
    attivita={}
    risorse={}
    attori={}
    a=[]
    r=[]
    att=[]
    for act in loadXES.get_sentences_XES(logName+'.xes'):
        a.append(act)
    for res in loadXES.get_resources_names(logName+'.xes'):
        r.append(res)
    for actor in loadXES.get_actors_names(logName+'.xes'):
        att.append(actor)
    attivita=list(sorted(dizionario(a).items(),key=itemgetter(1)))
    attivita.reverse()
    risorse=list(sorted(dizionarioX(r).items(),key=itemgetter(1)))
    risorse.reverse()
    attori=list(sorted(dizionarioX(att).items(),key=itemgetter(1)))
    attori.reverse()
    return attivita,risorse,attori


def auc_(path1, path2, lw=None):
    df = np.asarray(pd.read_csv(path2))
    dt = np.asarray(pd.read_csv(path1))
    X = {}
    y = {}
    for i in df:
        y[i[0]] = i[1]
    for j in dt:
        X[j[0]] = j[1]
    tmp1 = []
    tmp2 = []
    for key in X:
        tmp1.append(X[key])
    for key in y:
        tmp2.append(y[key])
    X = tmp1
    y = tmp2
    if (len(X) > len(y)):
        X = X[:len(y)]
    if (len(y) > len(X)):
        y = y[:len(X)]
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    X_train = np.asarray(X_train).reshape(-1, 1)
    X_test = np.asarray(X_test).reshape(-1, 1)
    classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    return str(roc_auc_score(y_test, y_score))


def getGetValues(path1, path2):
    return [NMIscore(path1, path2), Rscore(path1, path2), auc_(path1, path2)]


def plots(title, pathL2V, pathN2V, pathG2V):
    fig, ax = plt.subplots(figsize=(3 * len(pathL2V), 5))
    plt.grid()
    labels = []
    plt.title(title)
    for i in range(len(pathL2V)):
        labels.append(pathN2V[i])
        labels.append(pathG2V[i])

    ax.set_xticklabels([x.split("/")[-1] for x in labels], ha="right", size=8, rotation=8)
    markers = ["v", "^", "8"]
    color = ["blue", "green", "brown"]

    NMI = lines.Line2D([], [], color='blue', marker='v',
                       markersize=10, label='NMI SCORE')

    R = lines.Line2D([], [], color='green', marker='^',
                     markersize=10, label='R SCORE')

    AUC = lines.Line2D([], [], color='brown', marker='8',
                       markersize=10, label='AUC SCORE')

    plt.legend(handles=[NMI, R, AUC])
    for i in range(len(pathL2V)):

        values = getGetValues(pathL2V[i], pathN2V[i])

        for j in range(len(values)):
            ax.scatter(pathN2V[i], float(values[j]), marker=markers[j], s=150, color=color[j])

        values = getGetValues(pathL2V[i], pathG2V[i])

        for j in range(len(values)):
            ax.scatter(pathG2V[i], float(values[j]), marker=markers[j], s=150, color=color[j])
    plt.savefig(title + '.pdf')
    plt.show()


def generaPlot():
    logName = 'BPIC15GroundTruth'

    pathsL2VH=['./output/L2VVS16HierWard.csv','./output/L2VVS32HierWard.csv','./output/L2VVS64HierWard.csv','./output/L2VVS128HierWard.csv']
    pathsN2VH=['./output/N2VVS16HierWard.csv','./output/N2VVS32HierWard.csv','./output/N2VVS64HierWard.csv','./output/N2VVS128HierWard.csv']
    pathsG2VH=['./output/G2VVS16HierWard.csv','./output/G2VVS32HierWard.csv','./output/G2VVS64HierWard.csv','./output/G2VVS128HierWard.csv']

    # PLOT PER Hierward
    plots("PLOT PER HierWard", pathsL2VH,pathsN2VH,pathsG2VH)

    pathsL2VK=['./output/L2VVS16KMeans.csv','./output/L2VVS32KMeans.csv','./output/L2VVS64KMeans.csv','./output/L2VVS128KMeans.csv']
    pathsN2VK=['./output/N2VVS16KMeans.csv','./output/N2VVS32KMeans.csv','./output/N2VVS64KMeans.csv','./output/N2VVS128KMeans.csv']
    pathsG2VK=['./output/G2VVS16KMeans.csv','./output/G2VVS32KMeans.csv','./output/G2VVS64KMeans.csv','./output/G2VVS128KMeans.csv']


    plots("PLOT PER KMeans", pathsL2VK,pathsN2VK,pathsG2VK)


def test(path1,path2):
    print("Risultato di NMIScore: "+path1+' '+ NMIscore(path1,path2))
    print("Risultato di RScore: "+path1+' '+ Rscore(path1,path2))
    print("Risultato di AUC: "+path1 +' '+ AUC(path1,path2))

if __name__ == '__main__':
   logName = 'BPIC15GroundTruth'
   path1='./output/G2VVST32KMeans.csv'
   path2='./output/L2VVST32KMeans.csv'
   test(path1,path2)
   print("Genero i plot...")
   generaPlot()
   print("Elaboro info dal log...")
   attivita,risorse,attori=freqNumerica(logName)
   print("Attività con relativa frequenza:")
   print(attivita)
   print("Risorse con relativa frequenza:")
   print(risorse)
   print("Attori con relativa frequenza:")
   print(attori)



