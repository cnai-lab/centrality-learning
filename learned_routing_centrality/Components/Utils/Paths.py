import os


class ExpCentrality:
    def __init__(self, centrality, exp_num):
        self.n_routing_per_graph = 1
        self.root_path = os.path.join('..', '..', 'Experiments', 'LearningCentrality', f'Exp{exp_num}', f'{centrality}')
        self.train_graphs_path = os.path.join(self.root_path, 'train')
        self.validation_graphs_path = os.path.join(self.root_path, 'validation')
        self.test_graphs_path = os.path.join(self.root_path, 'test')
        self.description = f'Exp{exp_num}_{centrality}_Centrality'
