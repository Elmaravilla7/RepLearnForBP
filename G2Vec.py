import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import loadXES
import numpy as np

def learn(logName,vectorsize):
    learnT(logName,vectorsize)
    learnV(logName,vectorsize)

def learnT(logName,vectorsize):
    documents=buildGT(logName)
    model = gensim.models.Doc2Vec(documents, dm=0, alpha=0.025, vector_size=vectorsize, window=8, min_alpha=0.025,min_count=0, workers=4)
    nrEpochs=4
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(documents,total_examples=len(documents), epochs=nrEpochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    model.save('output/'+'G2VVST'+str(vectorsize)+'.model')

def learnV(logName,vectorsize):
    documents=buildG(logName)
    model = gensim.models.Doc2Vec(documents, dm=0, alpha=0.025, vector_size=vectorsize, window=8, min_alpha=0.025,min_count=0, workers=4)
    nrEpochs=4
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(documents,total_examples=len(documents), epochs=nrEpochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    model.save('output/'+'G2VVS'+str(vectorsize)+'.model')

def buildGT(logName):
    grafoS=[]
    attivita=[]
    for element in loadXES.get_sentences_XES(logName + ".xes"):
        attivita.append(pulisci(element))
    NodiAttivita={}
    i=0
    for trace in loadXES.get_trace_names(logName+".xes"):
            NodiAttivita[trace]=attivita[i]
            i=i+1
    for key in NodiAttivita:
        t=TaggedDocument(NodiAttivita[key],[key])
        grafoS.append(t)
    return grafoS

def buildG(logName):
    grafoS=[]
    attivita=[]
    for element in loadXES.Vget_sentences_XES(logName + ".xes"):
        attivita.append(element)
    NodiAttivita={}
    i=0
    for variant in loadXES.get_variant_names(logName+".xes"):
            NodiAttivita[variant]=attivita[i]
            i=i+1
    for key in NodiAttivita:
        t=TaggedDocument((np.concatenate(NodiAttivita[key])),[key])
        grafoS.append(t)
    return grafoS

def pulisci(element):
    attivita=[]
    for x in element:
        if(x!='ArtificialStartTask' and x!='ArtificialEndTask' and x not in attivita):
            attivita.append(x)
    return attivita

if __name__ == '__main__':
    logName = 'BPIC15GroundTruth'
    learn(logName,16)
    learn(logName,32)
    learn(logName,64)
    learn(logName,128)



