import torch
from torch import nn
from Components.RBC import RBC
from Components.Utils.CommonStr import EigenvectorMethod


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class LRCModel(nn.Module):
    def __init__(self, dim, device, dtype):
        super(LRCModel, self).__init__()
        self.stuv_dic= {}
        self.device, self.dtype, self.embed_dim = device, dtype, dim
        self.flatten = nn.Flatten()
        self.rbc_handler = RBC(eigenvector_method=EigenvectorMethod.torch_eig, pi_max_error=0.00001,
                               device=self.device, dtype=self.dtype)
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.conv_pool_stack = nn.Sequential(
            nn.Conv1d(4, 4, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device=device, dtype=dtype)
        self.conv_pool_stack.apply(init_weights)
    #direct model method
    def forward(self, X, expected_centrality_vectors=None):
        """
        Learning the routing function.
        :param X: the nodes embeding 
        :param expected_centrality_vectors:
        :return actual_rbc_vectors, pearson_scores: if expected_centrality_vectors is None return only actual_rbc_vectors
        """

        num_nodes_lst, node_embeddings_lst, edges_lst, sTBfs_lst = X
        actual_rbcs, pearson_scores = [], []

        if expected_centrality_vectors is not None:
            for i in range(len(num_nodes_lst)):
                rbc, pearson = self.compute_rbc(num_nodes_lst[i], node_embeddings_lst[i], edges_lst[i], sTBfs_lst[i],
                                                expected_centrality_vectors[i])
                actual_rbcs.append(rbc)
                pearson_scores.append(pearson)

            return actual_rbcs, pearson_scores
        else:
            for i in range(len(num_nodes_lst)):
                rbc = self.compute_rbc(num_nodes_lst[i], node_embeddings_lst[i], edges_lst[i], sTBfs_lst[i])
                actual_rbcs.append(rbc)
            return actual_rbcs
    def compute_rbc(self, num_nodes, node_embeddings, edges, sTBfs, expected_centrality_vec=None):
        """
        Computing actual RBC vector by summing delta_st
        :param num_nodes: graph's num nodes
        :param node_embeddings: graph's nodes embedding
        :param edges: graph's edges
        :param expected_centrality_vec: the expected centrality vector
        :return actual_rbc_vector, pearson_scores:
        """

        #uv_matrix = self.create_uv_matrix(node_embeddings, edges)
        u_embedding= [node_embeddings[u][None, :] for u, v in edges]
        u_embedding= torch.cat(u_embedding, dim=0)
        v_embedding= [node_embeddings[v][None, :] for u, v in edges]
        v_embedding= torch.cat(v_embedding, dim=0)
        u_embedding = u_embedding[:, None, :].to(device=self.device, dtype=self.dtype)
        v_embedding = v_embedding[:, None, :].to(device=self.device, dtype=self.dtype)
        actual_rbc = torch.zeros(num_nodes, device=self.device, dtype=self.dtype)
        ones_tensor = torch.zeros(size=(num_nodes, num_nodes), device=self.device, dtype=self.dtype)

        for s in range(num_nodes):
            s_embedding = node_embeddings[s]
            s_embedding = s_embedding.repeat(repeats=(len(edges), 1))
            s_embedding = s_embedding[:, None, :].to(device=self.device, dtype=self.dtype)
            ones_tensor[s][s] += 1
            for t in range(num_nodes):
                if s != t:  # when s==t delta_ss is zero
                    t_embedding = node_embeddings[t]
                    t_embedding = t_embedding.repeat(repeats=(len(edges), 1))
                    t_embedding = t_embedding[:, None, :].to(device=self.device, dtype=self.dtype)
                    stuv_embedding = torch.cat([s_embedding, t_embedding, u_embedding, v_embedding], dim=1)
                    stuv_predictions = self.conv_pool_stack(stuv_embedding).squeeze()
                    stuv_routing = torch.sparse_coo_tensor(torch.tensor(edges).t(), stuv_predictions,
                                                           (num_nodes, num_nodes), device=self.device,
                                                           dtype=self.dtype).to_dense()
                    delta_st = self.rbc_handler.delta_st(s, stuv_routing.clone() , T_val=1, t=t, routingBfs= sTBfs)
                    self.stuv_dic[s, t]= stuv_routing
                    actual_rbc += delta_st
            ones_tensor[s][s] -= 1
        actual_rbc = actual_rbc / torch.sum(actual_rbc)

        if expected_centrality_vec is not None:
            pearson_score = self.pearsonr(expected_centrality_vec.cuda(), actual_rbc)

            return actual_rbc, pearson_score
        else:
            return actual_rbc

    def create_uv_matrix(self, node_embeddings, edges):
        uv_pairs = [torch.cat([node_embeddings[u], node_embeddings[v]]).unsqueeze(dim=0) for u, v in edges]
        #zero-dimensional tensor (at position 0) cannot be concatenated
        uv_matrix = torch.cat(uv_pairs, dim=0)
        uv_matrix = uv_matrix.to(device=self.device, dtype=self.dtype)

        return uv_matrix

    def pearsonr(self, x, y):
        """
        https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739
        :param x: the learned rbc vector
        :param y: the expected centrality vector
        :return pearson: the pearson corr score between the vectors
        """
        pearson = self.cos(x - x.mean(dim=0, keepdim=True), y - y.mean())
        return pearson
