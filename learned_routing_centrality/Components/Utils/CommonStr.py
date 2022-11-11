import torch


class RbcMatrices:
    name = "RBC_Computing Matrices"
    root_path = "Root Path"
    adjacency_matrix = 'Adjacency Matrix'
    routing_policy = "Routing Policy"
    traffic_matrix = "Traffic Matrix"


class EmbeddingOutputs:
    graphs_root_path = "Graphs Path"
    trained_model_root_path = "Trained Model Path"
    trained_model = "Model"


class HyperParams:
    name = "Hyper Params"
    epochs = "Epochs"
    learning_rate = "Learning Rate"
    momentum = "Momentum"
    weight_decay = "Weight Decay"
    optimizer = "Optimizer"
    pi_max_err = "PI Max Error"
    error_type = "Error Type"
    batch_size = "Batch Size"


class EmbeddingStatistics:
    name = "EmbeddingsStatistics"
    id = "ID"
    centrality = "Centrality Type"
    centrality_params = "Centrality Params"
    n_graphs_train = "num graphs train"
    n_graphs_validation = "num graphs validation"
    n_graphs_test = "num graphs test"
    n_seeds_train_graph = "num seeds per train graph"
    n_routing_policy_per_graph = "num routing policy per train graph"
    n_random_samples_per_graph = 'num random samples graph'
    embd_dim = "Embedding Dimensions"
    pearson_weight = "Pearson Weight"
    centrality_weight = "Centrality Weight"
    rbcs_expected = "RBCs Expected"
    rbcs_actual = "RBCs Actual"
    euclidean_distance_median_train = "Euclidean Distance Median train"
    kendall_tau_b_avg_train = "KendallTau_b Avg train"
    pearson_avg_train = "Pearson Avg train"
    spearman_avg_train = "Spearman Avg train"
    euclidean_distance_median = "Euclidean Distance Median"
    kendall_tau_b_avg = "KendallTau_b Avg"
    pearson_avg = "Pearson Avg"
    spearman_avg = "Spearman Avg"
    train_error = "Train Error"
    validation_error = "Validation Error"
    error_type = "Error Type"
    network_structure = "Network Structure"
    train_runtime = "Train RunTime"
    embedding_alg = "Embedding Alg"
    learning_rate = HyperParams.learning_rate
    epochs = HyperParams.epochs
    weight_decay = HyperParams.weight_decay
    optimizer = HyperParams.optimizer
    batch_size = HyperParams.batch_size
    optimizer_params = "optimizer params"
    eigenvector_method = "Eigenvector Computing Method "
    pi_max_err = HyperParams.pi_max_err
    trained_model_path = "Trained Model Path"
    graphs_root_path = "Graphs Path"
    comments = "Comments"
    device = "Torch Device"
    dtype = "Torch Dtype"
    csv_save_path = "Saving Csv Path"
    graphs_desc = "Graph desc"
    with_ones = "with_ones"
    val_one_err = "val ones err"
    val_zeros_err = "val zeros err"
    val_rest_err = "val rest err"
    train_acc = "Train Acc"
    val_acc = "Val Acc"

    cols = [id, centrality, n_graphs_train, n_graphs_validation, n_graphs_test, n_seeds_train_graph,
            n_routing_policy_per_graph, graphs_desc, n_random_samples_per_graph, embedding_alg, embd_dim,
            pearson_weight,
            centrality_weight, rbcs_expected, rbcs_actual,  euclidean_distance_median_train, kendall_tau_b_avg_train,
            pearson_avg_train, spearman_avg_train, euclidean_distance_median, kendall_tau_b_avg, pearson_avg,
            spearman_avg, train_error, validation_error, error_type, network_structure, centrality_params,
            train_runtime, learning_rate, epochs, batch_size, weight_decay, optimizer, optimizer_params,
            eigenvector_method, pi_max_err, graphs_root_path, trained_model_path, comments, device, dtype, with_ones,
            train_acc, val_acc, val_one_err, val_zeros_err, val_rest_err]


class StatisticsParams:
    name = "Statistics Params"
    id = "ID"
    centrality = "Centrality Type"
    centrality_params = "Centrality Params"
    num_nodes = "Num Nodes"
    num_edges = "Num Edges"
    target = "Target"
    prediction = "Prediction"
    error = "Error"
    error_type = "Error Type"
    sigmoid = "With Sigmoid"
    src_src_one = "Predecessor[src, src]=1"
    src_row_zeros = "Predecessor[src, :]=0"
    target_col_zeros = "Predecessor[:, target]=0"
    fixed_R = "Fixed R policy"
    fixed_T = "Fixed T Matrix"
    runtime = "RunTime"
    learning_rate = HyperParams.learning_rate
    epochs = HyperParams.epochs
    weight_decay = HyperParams.weight_decay
    momentum = HyperParams.momentum
    optimizer = HyperParams.optimizer
    pi_max_err = HyperParams.pi_max_err

    path = "Path"
    comments = "Comments"
    eigenvector_method = "Eigenvector Computing Method "
    device = "Torch Device"
    dtype = "Torch Dtype"
    consider_traffic_paths = "Traffic Paths"
    optimizer_params = "optimizer params"
    csv_save_path = "Saving csv Path"

    cols = [id, centrality, centrality_params, num_nodes, num_edges, epochs, learning_rate, weight_decay,
            optimizer, optimizer_params, eigenvector_method, pi_max_err, sigmoid, src_src_one, src_row_zeros,
            target_col_zeros, fixed_T, fixed_R, consider_traffic_paths, device, dtype, path, comments, target
        , prediction, error, error_type, runtime]


class LearningParams:
    name = "Learning Params"
    hyper_parameters = "Hyper Parameters"
    adjacency_matrix = RbcMatrices.adjacency_matrix
    target = StatisticsParams.target
    src_src_one = StatisticsParams.src_src_one
    src_row_zeros = StatisticsParams.src_row_zeros
    target_col_zeros = StatisticsParams.target_col_zeros
    sigmoid = StatisticsParams.sigmoid
    fixed_R = StatisticsParams.fixed_R
    fixed_T = StatisticsParams.fixed_T
    eigenvector_method = StatisticsParams.eigenvector_method
    device = StatisticsParams.device
    dtype = StatisticsParams.dtype
    consider_traffic_paths = StatisticsParams.consider_traffic_paths
    centrality_params = StatisticsParams.centrality_params


class TorchDevice:
    cpu = torch.device('cpu')
    gpu = torch.device('cuda:0')


class TorchDtype:
    float = torch.float


class EigenvectorMethod:
    power_iteration = "Power Iteration"
    torch_eig = "Torch eig"


class OptimizerTypes:
    Rprop = "Rprop"
    RmsProp = "RmsProp"
    LBFGS = "LBFGS"
    AdaMax = "AdaMax"
    SparseAdam = "SparseAdam"
    AdaGrad = "AdaGrad"
    AdaDelta = "AdaDelta"
    Adam = "Adam"
    SGD = "SGD"
    ASGD = "ASGD"
    AdamW = "ADAMW"


class Centralities:
    name = "Centralities"
    SPBC = "Betweenness"
    CFBC = "Current Flow Betweenness"
    Closeness = "Closeness"
    CFC = "Current Flow Closeness"
    Load = "Load"
    Degree = "Degree"
    Eigenvector = "Eigenvector"
    centrality_lst = [SPBC, CFBC, Closeness, CFC, Load, Degree, Eigenvector]


class ErrorTypes:
    mse = "MSE"
    L1 = "L1Loss"
    SmoothL1 = "SmoothL1Loss"


class EmbeddingPathParams:
    seed = 'seed'


class EmbeddingAlgorithms:
    node2vec = "Node2Vec"
    node2vec_torch = "Node2VecTorch"
    diff2vec = "Diff2Vec"
    rand_ne = "RandNE"
    glee = "GLEE"
    net_mf = "NetMF"
    nnsed = "NNSED"
    danmf = "DANMF"
    mnmf = "MNMF"
    big_clam = "BigClam"
    symm_nmf = "SymmNMF"
    socio_dim = "SocioDim"
    node_sketch = "NodeSketch"
    boost_ne = "BoostNE"
    walklets = "Walklets"
    gra_rep = "GraRep"
    nmfadmm = "NMFADMM"
    laplacian_eigenmaps = "LaplacianEigenmaps"
    feather_node = "FeatherNode"
    ae = "AE"
    deep_walk = "DeepWalk"
    graph_wave = "GraphWave"
    musae = "MUSAE"
    role2vec = "Role2Vec"
    gl2vec = "GL2Vec"
    graph_2_vec = "Graph2Vec"


class Techniques:
    node_embedding_to_value = "node_embededding_value"
    node_embedding_s_t_routing = "node_embedding_s_t_routing"
    graph_embedding_to_routing = "graph_embedding_routing"
    node_embedding_to_routing = "node_embedding_routing"
    graph_embedding_to_rbc = "graph_embedding_rbc"
    optimize_centrality = "Optimize Centrality"
    optimize_st_routing = "Optimize st Routing"
    optimize_st_eig = "Optimize st Eig"


class RoutingTypes:
    fixed = "Fixed"
    similar = "Similar"


class NumRandomSamples:
    N = "O(N)"
    N_power_2 = "O(N^2)"
