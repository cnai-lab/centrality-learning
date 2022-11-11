from Components.Utils.CommonStr import HyperParams, LearningParams, EmbeddingStatistics, EmbeddingOutputs


def get_centrality_params(centrality, device, dtype):
    return ""


class ParamsManager:
    def __init__(self, params_dict):
        self.path_obj = params_dict['path_obj']
        self.exp_num = params_dict['exp_num']
        self.graphs_root_path = self.path_obj.root_path
        self.graphs_desc = self.path_obj.description
        self.centrality = params_dict[EmbeddingStatistics.centrality]
        self.embedding_dimensions = params_dict[EmbeddingStatistics.embd_dim]
        self.device = params_dict[EmbeddingStatistics.device]
        self.dtype = params_dict[EmbeddingStatistics.dtype]
        self.hyper_params = {
            HyperParams.learning_rate: params_dict[HyperParams.learning_rate],
            HyperParams.epochs: params_dict[HyperParams.epochs],
            HyperParams.momentum: params_dict[HyperParams.momentum],
            HyperParams.optimizer: params_dict[HyperParams.optimizer],
            HyperParams.batch_size: params_dict[HyperParams.batch_size],
            HyperParams.weight_decay: params_dict[HyperParams.weight_decay]
        }
        self.learning_params = {
            LearningParams.hyper_parameters: self.hyper_params,
            LearningParams.device: self.device,
            LearningParams.dtype: self.dtype
        }
