from networkx.algorithms.centrality.betweenness import _single_source_shortest_path_basic
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import random

def _subpolicy(S, P, sigma, graphSize):
    delta = np.zeros((graphSize, graphSize))
    policy = np.zeros((graphSize, graphSize, graphSize))
    while S:
        w = S.pop()
        temp1= np.zeros(graphSize)
        temp1[w]= 1.0
        coeff = (delta[w] + temp1) / sigma[w]
        for v in P[w]:
            policy[:, v, w]= sigma[v] * coeff
            delta[v] += sigma[v] * coeff
    policy=np.divide(policy, delta, out=np.zeros_like(policy), where=delta!=0)
    return policy#s,t,u,v



def srbc_policy(G):
    nodeSizeG=G.number_of_nodes()
    routingPolicy = np.zeros((nodeSizeG, nodeSizeG, nodeSizeG, nodeSizeG))# s, u, v, t
    routingBfs= np.zeros((nodeSizeG, nodeSizeG), dtype=int)
    for s in G:
        # use BFS
        S, P, sigma, _= _single_source_shortest_path_basic(G, s)# S: travel points, P: pre point, sigma how many messages received or transferred; all length is 150 and P is a 2 dimension list
        # accumulation
        P=list(P.values())
        sigma=list(sigma.values())
        routingBfs[s]= np.array(S, dtype=int)
        subPolicy= _subpolicy(S, P, sigma, nodeSizeG)
        routingPolicy[s]=subPolicy
    return routingPolicy, routingBfs


if __name__=="__main__":
    graphSize=50
    pos = {i: (random.gauss(0, 2), random.gauss(0, 2)) for i in range(70)}
    G=nx.random_geometric_graph(graphSize,radius=2, pos=pos)

    startT=time.time()
    routingPolicy=srbc_policy(G)
    print(time.time()-startT)

    nx.draw(G,nx.kamada_kawai_layout(G),with_labels = True, node_size=400, cmap=plt.cm.Blues)#spectral layout, eigenvectors of the graph Laplacian
    plt.show()