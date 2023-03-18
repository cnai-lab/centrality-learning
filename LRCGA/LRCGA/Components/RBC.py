from itertools import product
import torch
import numpy as np
import time

class RBC:
    def __init__(self, eigenvector_method, pi_max_error, device, dtype):
        self.eigenvector_method = eigenvector_method
        self.device = device
        self.dtype = dtype

    def compute_rbc_simple(self, n_nodes, R, T, Transform=True):
        """
        Computing Routing Betweenness Centrality of the given graph
        :param n_nodes: number of nodes
        :param R: routing function
        :param T: traffic matrix
        :return:
        """
        st_tuples = list(product(range(n_nodes), range(n_nodes)))
        all_delta_arrays = []
        sLast=0
        for s, t in st_tuples:
            if s!=sLast:
                sLast=s
            if s != t:
                R[s, t, s, s]=1.0
                all_delta_arrays.append(self.delta_st_simple(s, R[s, t], 1, t, Transform))
        rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)

        return rbc_arr

    def delta_st_simple(self, src, R_st, T_val, t, Transform=True):
        """
        :param src: source node
        :param R_st: routing function at s,t -> R(s,t)
        :param T_val: traffic volume from source node to target node
        :param t: target node
        :return normalized_eigenvector: the principal eigenvector of R(s,t)
        """
        if Transform:
            R_st = R_st.t()# nodes*nodes
        try:
            eigenvalues_lin, eigenvectors_lin = torch.linalg.eig(input=R_st)# use A square method
        except:
            print(R_st)
            exit(0)
        eigenvalues_lin, eigenvectors_lin = eigenvalues_lin.real, eigenvectors_lin.real
        eigenvector = eigenvectors_lin[:, torch.argmax(eigenvalues_lin)]
        normalized_eigenvector = self.normalize_eigenvector(src, eigenvector, T_val)
        return normalized_eigenvector
    def compute_rbc(self, n_nodes, R, T, routingBfs=None):
        """
        Computing Routing Betweenness Centrality of the given graph
        :param n_nodes: number of nodes
        :param R: routing function
        :param T: traffic matrix
        :return:
        """
        st_tuples = list(product(range(n_nodes), range(n_nodes)))
        all_delta_arrays = []
        sLast=0
        sourceTargetLength= None
        if routingBfs is not None and len(routingBfs.shape)==2:
            sourceTargetLength= np.zeros((n_nodes, n_nodes, 1), dtype=int)
        for s, t in st_tuples:
            if s!=sLast:
                sLast=s
            if s != t:
                all_delta_arrays.append(self.delta_st(s, R[s, t], 1, t, routingBfs, sourceTargetLength))
        rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)

        return rbc_arr, sourceTargetLength

    def delta_st(self, src, R_st, T_val, t, routingBfs= None, sourceTargetLength= None):
        """
        :param src: source node
        :param R_st: routing function at s,t -> R(s,t)
        :param T_val: traffic volume from source node to target node
        :param t: target node
        :return eigenvector: the principal eigenvector of R(s,t)
        """
        nodeNum= len(R_st)
        if len(routingBfs.shape)==2:
            R_st_c= R_st.clone()
            R_st_mul= R_st.clone()
            R_st=R_st[src]
            while R_st[t]==0:
                R_st_mul= torch.matmul(R_st_mul, R_st_c)
                R_st+= R_st_mul[src]
                sourceTargetLength[src, t, 0]+=1
            return R_st
        elif len(routingBfs.shape)==4:
            R_st_c= R_st.clone()
            R_st=R_st[src]
            sourceBfs= routingBfs[src]
            for i in range(nodeNum):
                midPoint= sourceBfs[i]
                R_st[midPoint]+= torch.matmul(R_st, R_st_c[:, midPoint])
                if midPoint== t:
                    return R_st
        elif len(routingBfs.shape)==3:
            R_st_c= R_st.clone()
            R_st_mul= R_st.clone()
            R_st=R_st[src]
            loopTimes=routingBfs[src, t, 0]
            for i in range(loopTimes):
                R_st_mul= torch.matmul(R_st_mul, R_st_c)
                R_st+= R_st_mul[src]
            return R_st


    def normalize_eigenvector(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of 1/eigenvector[src], zero
        n_eigenvector = eigenvector * x * T_val

        return n_eigenvector
