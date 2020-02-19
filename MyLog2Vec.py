import gensim
import loadXES
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import AgglomerativeClustering

def learn(logName,vectorsize):
    documents = loadXES.get_doc_XES_tagged(logName+'.xes')
    model = gensim.models.Doc2Vec(documents, dm = 0, alpha=0.025, vector_size= vectorsize, window=3, min_alpha=0.025, min_count=0)
    nrEpochs= 4
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(documents,total_examples=len(documents), epochs=nrEpochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    model.save('output/'+'L2VVS'+str(vectorsize) +'.model')

def cluster(logName,vectorsize,clusterType):
    clusterT(logName,vectorsize,clusterType)
    clusterV(logName,vectorsize,clusterType)

def clusterT(logName, vectorsize, clusterType):
    corpus = loadXES.get_doc_XES_tagged(logName+'.xes')
    model = gensim.models.Doc2Vec.load('output/' + 'L2VVS' + str(vectorsize) + '.model')
    vectors = []
    NUM_CLUSTERS= 5
    print("inferring vectors")
    for doc_id in range(len(corpus)):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        vectors.append(inferred_vector)
    print("done")
    if(clusterType=="KMeans"):
        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(vectors, assign_clusters=True)
    elif(clusterType=="HierWard"):
        ward = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, linkage='ward').fit(vectors)
        assigned_clusters = ward.labels_
    else:
        print(clusterType, " is not a predefined cluster type. Please use 'KMeans' or 'HierWard', or create a definition for ", clusterType)
        return
    trace_list = loadXES.get_trace_names(logName+".xes")
    clusterResult= {}
    for doc_id in range(len(corpus)):
        clusterResult[trace_list[doc_id]]=assigned_clusters[doc_id]
    resultFile= open('output/'+'L2VVST'+str(vectorsize)+clusterType+'.csv','w')
    for doc_id in range(len(corpus)):
        resultFile.write(trace_list[doc_id]+','+str(assigned_clusters[doc_id])+"\n")
    resultFile.close()
    print("done with " , clusterType , " on event log ", logName)

def clusterV(logName,vectorsize,clusterType):
    corpus = loadXES.get_doc_XES_tagged(logName+'.xes')
    vectors = []
    NUM_CLUSTERS = 5
    conta=[]
    model = gensim.models.Doc2Vec.load('output/' + 'L2VVS' + str(vectorsize) + '.model')
    print("inferring vectors")
    for variant in range(len(corpus)):
        if (corpus[variant].words not in conta):
            inferred_vector = model.infer_vector(corpus[variant].words)
            vectors.append(inferred_vector)
            conta.append(corpus[variant].words)
    print("done")
    if(clusterType=="KMeans"):
        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(vectors, assign_clusters=True)
    elif(clusterType=="HierWard"):
        ward = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, linkage='ward').fit(vectors)
        assigned_clusters = ward.labels_
    else:
        print(clusterType, " is not a predefined cluster type. Please use 'KMeans' or 'HierWard', or create a definition for ", clusterType)
        return
    clusterResult= {}
    variant_list = loadXES.get_variant_names(logName + ".xes")
    fatte = []
    for variant in range(len(conta)):
        if (variant_list[variant] not in fatte):
            clusterResult[variant_list[variant]] = assigned_clusters[variant]
            fatte.append(variant_list[variant])
    resultFile = open('output/' + 'L2VVS' + str(vectorsize) + clusterType + '.csv', 'w')
    fatte = []
    for variant in range(len(conta)):
        if (variant_list[variant] not in fatte):
            resultFile.write(variant_list[variant] + ',' + str(assigned_clusters[variant]) + "\n")
            fatte.append(variant_list[variant])
    resultFile.close()
    print('done')

if __name__ == '__main__':
    logName = 'BPIC15GroundTruth'

    #learn(logName,16)
    #learn(logName, 32)
    #learn(logName, 64)
    #learn(logName, 128)
    '''
    cluster(logName,16,"KMeans")
    cluster(logName, 32, "KMeans")
    cluster(logName, 64, "KMeans")
    cluster(logName, 128, "KMeans")
    '''
    cluster(logName,16,"HierWard")
    '''
    cluster(logName, 32, "HierWard")
    cluster(logName, 64, "HierWard")
    cluster(logName, 128, "HierWard")
    '''



