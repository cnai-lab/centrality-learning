import torch
from Components.Utils.CommonStr import OptimizerTypes


class Optimizer:
    def __init__(self, model, name, learning_rate, momentum=0, weight_decay=0):

        if name == OptimizerTypes.AdaDelta:
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif name == OptimizerTypes.AdaGrad:
            self.optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif name == OptimizerTypes.Adam:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif name == OptimizerTypes.AdamW:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif name == OptimizerTypes.SparseAdam:
            self.optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
        elif name == OptimizerTypes.AdaMax:
            self.optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif name == OptimizerTypes.ASGD:
            self.optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif name == OptimizerTypes.LBFGS:
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
        elif name == OptimizerTypes.RmsProp:
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, eps=1e-05,
                                                 centered=True, weight_decay=weight_decay)
        elif name == OptimizerTypes.Rprop:
            self.optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
        elif name == OptimizerTypes.SGD:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                                             weight_decay=weight_decay)

    def get_optimizer_params(self):
        all_optimizer_params = self.optimizer.param_groups[0]
        all_optimizer_params_keys = set(all_optimizer_params.keys())
        exclude_keys = {'params'}
        optimizer_params = {key: all_optimizer_params[key] for key in
                            all_optimizer_params_keys.difference(exclude_keys)}
        return optimizer_params

    def step(self):
        return self.optimizer.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def change_learning_rate(self, new_lr):
        self.optimizer.param_groups[0]['lr'] = new_lr