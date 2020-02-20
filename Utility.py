from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def test(path1,path2):
    print("Risultato di NMIScore: "+path1+' '+ NMIscore(path1,path2))
    print("Risultato di RScore: "+path1+' '+ Rscore(path1,path2))
    print("Risultato di AUC: "+path1 +' '+ AUC(path1,path2))

if __name__ == '__main__':
   logName = 'BPIC15GroundTruth'
   path1='./output/G2VVST32KMeans.csv'
   path2='./output/L2VVST32KMeans.csv'
   test(path1,path2)
   print("Elaboro info dal log...")
   attivita,risorse,attori=freqNumerica(logName)
   print("Attività con relativa frequenza:")
   print(attivita)
   print("Risorse con relativa frequenza:")
   print(risorse)
   print("Attori con relativa frequenza:")
   print(attori)



