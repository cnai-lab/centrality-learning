from routing_policy.srbc import srbc_policy
import networkx as nx
from Components.RBC import RBC
import torch 
import os
import random
import numpy as np
from torch import nn
import auralization

#os.chdir("/home/lx/code/1_2_centrality-learning-main/auralized_learned_routing_centrality")

def __pearsonr( x, y):
    """
    https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739
    :param x: the learned rbc vector
    :param y: the expected centrality vector
    :return pearson: the pearson corr score between the vectors
    """
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    pearson = cos(x - x.mean(dim=0, keepdim=True), y - y.mean())
    return pearson

def create_set(rootDir, indexesOfset=[1, 2], graphSize=50, radius=0.4, centralityName= "Betweenness", graphType="RG"):
    '''
    graphType:
        "RG", "BA", "WS", "ER"
    radius:
        only useful for random geometry graph 
    '''
    if centralityName=="Betweenness":
        centrality=nx.betweenness_centrality
    elif centralityName=="Closeness":
        centrality=nx.closeness_centrality
    elif centralityName=="Load":
        centrality=nx.load_centrality
    elif centralityName=="Eigenvector":
        centrality=nx.eigenvector_centrality
    elif centralityName=="Degree":
        centrality=nx.degree_centrality
    
    for i in indexesOfset:
        ID=i
        G= None
        if graphType=="RG":
            pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(graphSize)}
            G=nx.random_geometric_graph(graphSize,radius, pos=pos)
            #to make sure it is a connected graph
            while nx.is_connected(G)==False:
                pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(graphSize)}
                G=nx.random_geometric_graph(graphSize,radius, pos=pos)
        elif graphType=="BA":
            G=nx.barabasi_albert_graph(graphSize, 3)
            while nx.is_connected(G)==False:
                G=nx.barabasi_albert_graph(graphSize, 3)
        elif graphType=="WS":
            G=nx.watts_strogatz_graph(graphSize, 6, 0.7)
            while nx.is_connected(G)==False:
                G=nx.watts_strogatz_graph(graphSize, 6, 0.7)
        elif graphType=="ER":
            G=nx.erdos_renyi_graph(graphSize, 0.2)
            while nx.is_connected(G)==False:
                G=nx.erdos_renyi_graph(graphSize, 0.2)
        edgeList= np.array(G.edges())
        singleEdgeList= np.zeros((len(G.edges())*2, 2))
        singleEdgeList[0:len(G.edges()),:]=edgeList
        singleEdgeList[len(G.edges()): len(G.edges())*2 ,0]=edgeList[:,1]
        singleEdgeList[len(G.edges()): len(G.edges())*2 ,1]=edgeList[:,0]
        Rcc, routingBfs= srbc_policy(G)
        RccTensor= torch.tensor(Rcc, dtype=torch.float64)
        rbcHandle= RBC("srbc", "0", "RTX2050", float)
        _ , sourceTargetBfs= rbcHandle.compute_rbc(graphSize, RccTensor.clone(), None, routingBfs)
        
        if centrality==nx.eigenvector_centrality:
            rbcVector= centrality(G, 500)
        else: 
            rbcVector= centrality(G)
        rbcVector= torch.tensor(list(rbcVector.values()),  dtype=torch.float)

        nNode= torch.tensor([graphSize])
        nodesEmbeddings=auralization.get_auralization(G)
        nodesEmbeddings= torch.tensor(nodesEmbeddings, dtype=torch.float)
        sourceTargetBfs= torch.tensor(sourceTargetBfs, dtype=torch.int32)
        

        path= rootDir+ f"/net{ID}/"
        os.makedirs(path, exist_ok = True)
        torch.save(rbcVector, path+"expected_rbc.pt")#centrality vector
        torch.save(nNode, path+"n_nodes.pt")
        torch.save(nodesEmbeddings, path+"nodes_embeddings.pt")#v
        torch.save(sourceTargetBfs, path+"sourceTargetBfs.pt")#v
        np.savetxt(path+"edges.txt", singleEdgeList, fmt='%d', delimiter=",")#v


#using example
if __name__=="__main__":
    create_set("datas/degree/test", [i for i in range(2)], 30, radius=0.4, centralityName="Betweenness", graphType="BA")




