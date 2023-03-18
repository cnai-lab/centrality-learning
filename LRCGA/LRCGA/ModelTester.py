import os
print(os.getcwd())
import random
import torch
import numpy as np
import scipy.stats
import scipy.spatial.distance
from torch.utils.data import DataLoader
from Components.LRC.ModelHandler import train_model
from Components.LRC.LRCNN import LRCModel
import Components.Utils.Paths as Paths
from Components.LRC.Loaders import LRCLoader
from Components.LRC.Loaders.LRCLoader import CustomDatasetLRCLoader
from Components.Utils.Optimizer import Optimizer
from Components.Utils.ParamsManager import ParamsManager
from Components.Utils.CommonStr import EigenvectorMethod, EmbeddingStatistics, Centralities, TorchDevice, TorchDtype, \
    HyperParams, OptimizerTypes
import generating_data
import networkx as nx
import datetime

def run_test(pr_st):
    """
    The main flow
    :param pr_st: dictionary with all parameters required for the current run
    """

    # init
    set_seed()
    nn_model = init_nn_model(pr_st)
    optimizer = init_optimizer(pr_st, nn_model)
    params_man = ParamsManager(pr_st)
    start_train_time = datetime.datetime.now()
    # reload model 
    if pr_st[HyperParams.reload_weight]:
        try:
            checkpoint = torch.load("convergence_model/geometry/load/model_last.pt")#closeness, load
            nn_model.load_state_dict(checkpoint['model_state_dict'])
            print("get one model saved before ...")
            print(f"Last test correlation: {checkpoint['train_covariance']}, last train correlation: {checkpoint['test_covariance']}")
        except:
            print("no model saved before ...")
    # training
    end_flag= 0
    refreshNumber= int(pr_st[HyperParams.train_graph_num]*pr_st[HyperParams.train_graph_refresh_portion])
    generating_data.create_set(pr_st['path_obj'].train_graphs_path, [i for i in range(pr_st[HyperParams.train_graph_num])],
                                pr_st[HyperParams.graph_node_num], radius=0.4, centralityName=pr_st[EmbeddingStatistics.centrality],
                                  graphType=pr_st[HyperParams.graph_type])
    generating_data.create_set(pr_st['path_obj'].test_graphs_path, [i for i in range(pr_st[HyperParams.test_graph_num])],
                                pr_st[HyperParams.graph_node_num], radius=0.4, centralityName=pr_st[EmbeddingStatistics.centrality],
                                  graphType=pr_st[HyperParams.graph_type])
    refreshDataTime= params_man.hyper_params[HyperParams.epochs]//pr_st[HyperParams.train_graph_refresh_interval]
    for i in range(0, refreshDataTime):
        # get data loaders
        train_ldr, val_ldr, test_ldr = get_data_loaders(params_man)
        startEpoch= i*pr_st[HyperParams.train_graph_refresh_interval]
        endEpoch= (i+1)*pr_st[HyperParams.train_graph_refresh_interval]
        train_res = train_model(nn_model, optimizer, params_man, train_ldr, val_ldr, test_ldr, startEpoch, endEpoch, start_train_time)
        trained_model, train_err, validation_err, train_time = train_res
        if train_err<0.1:
            end_flag+=pr_st[HyperParams.train_graph_refresh_interval]
            if end_flag> 100:
                exit("End with endProcess!")
        randomlist = random.sample(range(0, pr_st[HyperParams.train_graph_num]), refreshNumber)
        generating_data.create_set(pr_st['path_obj'].train_graphs_path, randomlist,
                                    pr_st[HyperParams.graph_node_num], radius=0.4, centralityName=pr_st[EmbeddingStatistics.centrality],
                                      graphType=pr_st[HyperParams.graph_type])
        print(f"Generate new data: {randomlist}")

    # computing avg correlation scores after training 
    compute_avg_corr_scores(trained_model, train_ldr, is_test_ldr=False)
    compute_avg_corr_scores(trained_model, test_ldr, is_test_ldr=True)

    # clean mem
    del nn_model
    torch.cuda.empty_cache()


def init_optimizer(pr_st, nn_model):
    optimizer_name, lr, momentum = pr_st[EmbeddingStatistics.optimizer], pr_st[EmbeddingStatistics.learning_rate], \
                                   pr_st[HyperParams.momentum]
    wd = pr_st[HyperParams.weight_decay]
    optimizer = Optimizer(model=nn_model, name=optimizer_name, learning_rate=lr, momentum=momentum, weight_decay=wd)

    return optimizer


def get_data_loaders(p_man: ParamsManager):
    """
    Init DataLoaders for training/testing set
    :param p_man:
    :return:
    """

    g = torch.Generator()
    g.manual_seed(0)

    params = {'batch_size': p_man.hyper_params[HyperParams.batch_size],
              'shuffle': True,
              'collate_fn': LRCLoader.custom_collate_fn}
    
    train_paths, validation_paths, test_paths = get_tvt_paths()
    print("start...")
    train_set = CustomDatasetLRCLoader(train_paths, p_man.dtype, p_man.device)
    train_ldr = DataLoader(train_set, **params)

    test_set = CustomDatasetLRCLoader(test_paths, p_man.dtype, p_man.device)
    test_ldr = DataLoader(test_set, **params)

    if len(validation_paths) > 0:
        validation_set = CustomDatasetLRCLoader(validation_paths, p_man.dtype, p_man.device)
        validation_ldr = DataLoader(validation_set, **params)
    else:
        validation_ldr = DataLoader(test_set, **params)

    return train_ldr, validation_ldr, test_ldr


def get_tvt_paths():
    train_paths = [path[0] for path in list(os.walk(path_obj.train_graphs_path))[1:]]
    validation_paths = [path[0] for path in list(os.walk(path_obj.validation_graphs_path))[1:]]
    test_paths = [path[0] for path in list(os.walk(path_obj.test_graphs_path))[1:]]
    return train_paths, validation_paths, test_paths


def init_nn_model(param_embed):
    """
    initialize the Neural Network
    :param param_embed:
    :return:
    """
    device, dtype = param_embed[EmbeddingStatistics.device], param_embed[EmbeddingStatistics.dtype]
    embed_dimension = param_embed[EmbeddingStatistics.embd_dim]
    model = LRCModel(embed_dimension, device, dtype)

    return model


def set_seed(seed=44):
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # Python
    # if use_cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu vars
    torch.backends.cudnn.deterministic = True  # needed
    torch.backends.cudnn.benchmark = False


def compute_avg_corr_scores(model, ldr, is_test_ldr=False):
    """
    Computing correlation scores for Kendall, Pearson and Sperman between the learned to actual centrality vector
    :return:
    """
    all_expected, all_actual = [], []
    model.eval()
    with torch.no_grad():
        n_instances = ldr.dataset.__len__()
        dist_avg, kendall_avg, pearson_avg, spearman_avg = 0, 0, 0, 0
        for i, samples in enumerate(ldr):
            Xs, ys = samples[0], samples[1]
            actual_rbcs, expected_rbcs = model(Xs), ys
            if is_test_ldr:
                all_expected += expected_rbcs
                all_actual += actual_rbcs
            e_a = list(zip(expected_rbcs, actual_rbcs))

            euclidean_arr = np.array(
                [scipy.spatial.distance.euclidean(expected, actual.cpu()) for expected, actual in e_a])
            kendall_arr = np.array([scipy.stats.kendalltau(expected, actual.cpu())[0] for expected, actual in e_a])
            spearman_arr = np.array([scipy.stats.spearmanr(expected, actual.cpu())[0] for expected, actual in e_a])
            pearsonr_arr = np.array([scipy.stats.pearsonr(expected, actual.cpu())[0] for expected, actual in e_a])

            dist_avg += (euclidean_arr / n_instances).sum()
            kendall_avg += (kendall_arr / n_instances).sum()
            pearson_avg += (pearsonr_arr / n_instances).sum()
            spearman_avg += (spearman_arr / n_instances).sum()

        if is_test_ldr:
            print(f'\n--FINAL test set corr scores-- \n kendall: {kendall_avg} \n spearman: {spearman_avg} \n pearsonr: {pearson_avg}')
        else:
            print(f'\n--FINAL train set corr scores-- \n kendall: {kendall_avg} \n spearman: {spearman_avg} \n pearsonr: {pearson_avg}')


if __name__ == '__main__':
    experiments = [35]
    for exp_number in experiments:
        exp_num = exp_number    
        centrality = Centralities.SPBC
        path_obj = Paths.ExpCentrality(centrality, exp_num)

        params_statistics1 = {
            'exp_num': exp_num,
            EmbeddingStatistics.centrality: centrality,
            EmbeddingStatistics.embd_dim: 2,
            HyperParams.optimizer: OptimizerTypes.Adam,
            HyperParams.learning_rate: 3e-5,
            HyperParams.epochs: 1500,
            HyperParams.batch_size: 4,
            HyperParams.weight_decay: 0.0,
            HyperParams.momentum: 0.0,
            HyperParams.reload_weight: True,
            HyperParams.graph_node_num: 30,
            HyperParams.train_graph_num: 100,
            HyperParams.test_graph_num: 8,
            HyperParams.train_graph_refresh_portion: 0.5,
            HyperParams.train_graph_refresh_interval: 5,
            HyperParams.graph_type: "RG", #"RG", "BA", "WS", "ER"
            'path_obj': path_obj,
            EmbeddingStatistics.device: TorchDevice.gpu,
            EmbeddingStatistics.dtype: TorchDtype.float,
            EmbeddingStatistics.eigenvector_method: EigenvectorMethod.torch_eig
        }

        run_test(pr_st=params_statistics1)
