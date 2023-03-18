import os
from Components.Utils.CommonStr import Centralities

class ExpCentrality:
    def __init__(self, centrality, exp_num):
        self.n_routing_per_graph = 1
        self.root_path = os.path.join('datas'+str(exp_num), f'{centrality}')
        self.train_graphs_path = os.path.join(self.root_path, 'train')
        self.validation_graphs_path = os.path.join(self.root_path, 'validation')
        self.test_graphs_path = os.path.join(self.root_path, 'test')
        self.description = f'Exp{exp_num}_{centrality}_Centrality'
        print(self.test_graphs_path)
        print(self.train_graphs_path)

if __name__=="__main__":
    exp_num = [35]
    centrality = Centralities.SPBC
    path_obj = ExpCentrality(centrality, exp_num)
    pass