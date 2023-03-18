import networkx as nx
import numpy as np
import random



def get_auralization(G, momentum=0.9, SampleLength=100, Epsilon=1E-32):
    GraphSize=G.number_of_nodes()
    if nx.is_connected(G)==False:
        exit("Graph is not connected !!!")
    AdjacencyMatrix=nx.adjacency_matrix(G).todense()#<class 'numpy.matrix'>
    SMatrix=np.zeros((GraphSize,SampleLength))# create S matrix storing waveform
    SMatrix[:,0]=1# initialize the first state to 1
    DeltaSMatrix_t_1=np.zeros((GraphSize,GraphSize))
    PMatrix=(AdjacencyMatrix/AdjacencyMatrix.sum(axis=0)+Epsilon).transpose()#if denominator is a vector, matrix divided by a vector is row/column by each element
    #axis means which dimension to keep, 0 is row, so this is a sum of column
    for  i in range(1,SampleLength):
        tempZero=np.zeros((GraphSize,GraphSize))
        np.fill_diagonal(tempZero,SMatrix[:,i-1])
        DeltaSMatrix_t=tempZero.dot(PMatrix)+DeltaSMatrix_t_1*momentum
        DeltaSMatrix_t_1=DeltaSMatrix_t
        SMatrix[:,i]=SMatrix[:,i-1]+DeltaSMatrix_t.sum(axis=0).reshape(GraphSize)-DeltaSMatrix_t.sum(axis=1).reshape(GraphSize)
    meanValue=SMatrix.mean(axis=1)
    SMatrix=(SMatrix.transpose()-meanValue).transpose()
    return SMatrix


if __name__=="__main__":
    graphSize= 40
    radius= 0.4
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(graphSize)}
    G=nx.random_geometric_graph(graphSize,radius, pos=pos)
    #to make sure it is a connected graph
    while nx.is_connected(G)==False:
        pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(graphSize)}
        G=nx.random_geometric_graph(graphSize,radius, pos=pos)
    SMatrix=get_auralization(G)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(SMatrix[0,:])
    plt.show()



