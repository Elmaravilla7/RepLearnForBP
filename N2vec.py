from node2vec import Node2Vec
import networkx as nx
import loadXES

def learn(logName,vectorsize):
    learnT(logName,vectorsize)
    learnV(logName,vectorsize)

def learnT(logName,vectorsize):
    graph=nx.DiGraph()
    buildGT(graph,logName)
    node2vec = Node2Vec(graph, dimensions=vectorsize, walk_length=30, num_walks=200,workers=8)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.save('output/'+'N2VVST'+str(vectorsize)+'.model')

def learnV(logName,vectorsize):
    graph=nx.DiGraph()
    buildG(graph,logName)
    node2vec = Node2Vec(graph, dimensions=16, walk_length=30, num_walks=200,workers=8)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.save('output/'+'N2VVS'+str(vectorsize)+'.model')

def buildGT(graph,logName):
    max = 5000
    attivita=[]
    for element in loadXES.get_sentences_XES(logName + ".xes"):
        attivita.append(pulisci(element))
    NodiAttivita={}
    i=0
    c=0
    for trace in loadXES.get_trace_names(logName+".xes"):
        if(c<=max):
            NodiAttivita[trace]=attivita[i]
            i=i+1
            c=c+1
        else:
            break
    for key in NodiAttivita.keys():
            graph.add_node(key)
    for trac1 in NodiAttivita.keys():
        for trac2 in NodiAttivita.keys():
            if (trac1 != trac2):
                x = NodiAttivita[trac1]
                y = NodiAttivita[trac2]
                for el in x:
                    if (c < max):
                        if (y.__contains__(el)):
                            graph.add_edge(trac1, trac2, attr=el)
                            c = c + 1
                            break
                    else:
                        break

def buildG(graph,logName):
    max=5000
    attivita=[]
    for element in loadXES.Vget_sentences_XES(logName + ".xes"):
        attivita.append(element)
    NodiAttivita={}
    i=0
    c=0
    for variant in loadXES.get_variant_names(logName+".xes"):
        if(c<=max):
            NodiAttivita[variant]=attivita[i]
            i=i+1
            c=c+1
        else:
            break
    for key in NodiAttivita.keys():
        graph.add_node(key)
    for var1 in NodiAttivita.keys():
        for var2 in NodiAttivita.keys():
            if(var1!=var2):
                x = NodiAttivita[var1]
                y = NodiAttivita[var2]
                for el in x:
                    if(c<max):
                        if(y.__contains__(el)):
                            graph.add_edge(var1,var2,attr=el)
                            c=c+1
                            break
                    else:
                        break

def pulisci(element):
    attivita=[]
    for x in element:
        if(x!='ArtificialStartTask' and x!='ArtificialEndTask' and x not in attivita):
            attivita.append(x)
    return attivita

if __name__ == '__main__':
    logName = 'BPIC15GroundTruth'
    learn(logName,16)
    #learn(logName, 32)
    #learn(logName, 64)
    #learn(logName, 128)




