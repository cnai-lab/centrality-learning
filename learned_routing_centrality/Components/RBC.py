from itertools import product
import torch


class RBC:
    def __init__(self, eigenvector_method, pi_max_error, device, dtype):
        self.eigenvector_method = eigenvector_method
        self.device = device
        self.dtype = dtype

    def compute_rbc(self, n_nodes, R, T):
        """
        Computing Routing Betweenness Centrality of the given graph
        :param n_nodes: number of nodes
        :param R: routing function
        :param T: traffic matrix
        :return:
        """
        st_tuples = list(product(range(n_nodes), range(n_nodes)))
        all_delta_arrays = []
        for s, t in st_tuples:
            if s != t:
                all_delta_arrays.append(self.delta_st(s, R[s, t], 1, t))
        rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)

        return rbc_arr

    def delta_st(self, src, R_st, T_val, t):
        """
        :param src: source node
        :param R_st: routing function at s,t -> R(s,t)
        :param T_val: traffic volume from source node to target node
        :param t: target node
        :return normalized_eigenvector: the principal eigenvector of R(s,t)
        """
        R_st = R_st.t()
        eigenvalues_lin, eigenvectors_lin = torch.linalg.eig(input=R_st)
        eigenvalues_lin, eigenvectors_lin = eigenvalues_lin.real, eigenvectors_lin.real
        eigenvector = eigenvectors_lin[:, torch.argmax(eigenvalues_lin)]
        normalized_eigenvector = self.normalize_eigenvector(src, eigenvector, T_val)

        return normalized_eigenvector

    def normalize_eigenvector(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
        n_eigenvector = eigenvector * x * T_val

        return n_eigenvector
