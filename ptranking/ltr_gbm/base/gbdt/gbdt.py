
import pickle
import numpy as np

import torch

from ptranking.ltr_gbm.base.gbm.gbm import GBM

##- GBDT-specific torch.jit.script -##
@torch.jit.script
# copy from reference project, and customize to gbdt
def _leaf_prediction_gbdt(gradient: torch.Tensor, hessian: torch.Tensor, node: torch.Tensor,
                     estimator: int, reg_lambda: torch.Tensor, leaves_idx: torch.Tensor,
                     leaves_mu: torch.Tensor,
                     leaf_idx: int, split_node: torch.Tensor, yhat_train: torch.Tensor,
                     lr: torch.Tensor):
    gradient_sum = gradient.sum()
    hessian_sum = hessian.sum()

    # TODO-note: negative symbol (according to the GBDT-related equation)
    mu = - gradient_sum / (hessian_sum + reg_lambda)

    # Save optimal prediction and node information
    leaves_idx[estimator, leaf_idx] = node
    leaves_mu[estimator, leaf_idx] = mu
    yhat_train[split_node] -= lr * mu
    # Increase leaf idx
    leaf_idx += 1

    return leaves_idx, leaves_mu, leaf_idx, yhat_train

@torch.jit.script
# copy from reference project, and customize to gbdt
def _leaf_prediction_mu_gbdt(gradient: torch.Tensor, hessian: torch.Tensor, reg_lambda: torch.Tensor):
    gradient_sum = gradient.sum()
    hessian_sum = hessian.sum()

    # TODO-note: negative symbol (according to the GBDT-related equation)?
    mu = - gradient_sum / (hessian_sum + reg_lambda)
    return mu

@torch.jit.script
# copy from reference project, and customize to gbdt
def _create_tree_gbdt(X: torch.Tensor, gradient: torch.Tensor, hessian: torch.Tensor,
                 estimator: int, train_nodes: torch.Tensor, nodes_idx: torch.Tensor,
                 nodes_split_bin: torch.Tensor, nodes_split_feature: torch.Tensor,
                 leaves_idx: torch.Tensor, leaves_mu: torch.Tensor,
                 feature_importance: torch.Tensor,
                 yhat_train: torch.Tensor, learning_rate: torch.Tensor,
                 max_nodes: int, samples: torch.Tensor,
                 sample_features: torch.Tensor, max_bin: int,
                 min_data_in_leaf: torch.Tensor, reg_lambda: torch.Tensor,
                 min_split_gain: torch.Tensor, any_monotone: torch.Tensor,
                 monotone_constraints: torch.Tensor, monotone_iterations: int):
    # Set start node and start leaf index
    leaf_idx = 0
    node_idx = 0
    node = torch.tensor(1, dtype=torch.int64, device=X.device)
    next_nodes = torch.zeros((max_nodes * 2 + 1), dtype=torch.int64, device=X.device)
    next_node_idx = 0
    # Set constraint matrices for monotone constraints
    node_constraint_idx = 0
    node_constraints = torch.zeros((max_nodes * 2 + 1, 3), dtype=torch.float32, device=X.device)
    node_constraints[0, 0] = node
    node_constraints[:, 1] = -np.inf
    node_constraints[:, 2] = np.inf
    # Choose random subset of features
    Xe = X[sample_features]
    # Set other initial variables
    n_samples = samples.sum()
    grad_hess = torch.cat((gradient.unsqueeze(1), hessian.unsqueeze(1)), dim=1)
    # Create tree
    while (leaf_idx < max_nodes + 1) and (node != 0):
        # Retrieve samples in current node, and for train only the samples in the bagging batch
        split_node = train_nodes == node
        split_node_train = split_node * samples
        # Continue making splits until we exceed max_nodes, after that create leaves only
        if node_idx < max_nodes:
            # Select samples in current node
            X_node = Xe[:, split_node_train]
            grad_hess_node = grad_hess[split_node_train]
            # Compute split gain histogram
            Gl, Hl, Glc = torch.ops.pgbm.split_gain(X_node, grad_hess_node, max_bin)
            # Compute counts of right leaves
            Grc = grad_hess_node.shape[0] - Glc;
            # Sum gradients and hessian of the node
            G, H = grad_hess_node.sum(0).chunk(2, dim=0)
            # Compute total split_gain
            split_gain_tot = (Gl * Gl) / (Hl + reg_lambda) + \
                             (G - Gl) * (G - Gl) / (H - Hl + reg_lambda) - \
                             (G * G) / (H + reg_lambda)
            # Only consider split gain when enough samples in leaves.
            split_gain_tot *= (Glc >= min_data_in_leaf) * (Grc >= min_data_in_leaf)
            split_gain = split_gain_tot.max()
            # Split if split_gain exceeds minimum
            if split_gain > min_split_gain:
                argmaxsg = split_gain_tot.argmax()
                split_feature_sample = torch.div(argmaxsg, max_bin, rounding_mode='floor')
                split_bin = argmaxsg - split_feature_sample * max_bin
                split_left = (Xe[split_feature_sample] > split_bin).squeeze()
                split_right = ~split_left * split_node
                split_left *= split_node
                # Check for monotone constraints if applicable
                if any_monotone:
                    split_gain_tot_flat = split_gain_tot.flatten()
                    # Find min and max for leaf (mu) weights of current node
                    node_min = node_constraints[node_constraints[:, 0] == node, 1].squeeze()
                    node_max = node_constraints[node_constraints[:, 0] == node, 2].squeeze()
                    # Check if current split proposal has a monotonicity constraint
                    split_constraint = monotone_constraints[sample_features[split_feature_sample]]
                    # Perform check only if parent node has a constraint or if the current proposal is constrained. Todo: this might be a CPU check due to np.inf. Replace np.inf by: torch.tensor(float("Inf"), dtype=torch.float32, device=X.device)
                    if (node_min > -np.inf) or (node_max < np.inf) or (split_constraint != 0):
                        # We precompute the child left- and right weights and evaluate whether they satisfy the constraints. If not, we seek another split and repeat.
                        mu_left = _leaf_prediction_mu_gbdt(gradient[split_left], hessian[split_left], reg_lambda)
                        mu_right = _leaf_prediction_mu_gbdt(gradient[split_right], hessian[split_right], reg_lambda)
                        split = 1
                        split_iters = 1
                        condition = split * (((mu_left < node_min) + (mu_left > node_max) + \
                                              (mu_right < node_min) + (mu_right > node_max)) + \
                                             ((split_constraint != 0) * (
                                                         torch.sign(mu_right - mu_left) != split_constraint)))
                        while condition > 0:
                            # Set gain of current split to -1, as this split is not allowed
                            split_gain_tot_flat[argmaxsg] = -1
                            # Get new split. Check if split_gain is still sufficient, because we might end up with having only constraint invalid splits (i.e. all split_gain <= 0).
                            split_gain = split_gain_tot_flat.max()
                            # Check if new proposed split is allowed, otherwise break loop
                            split = (split_gain > min_split_gain) * int(split_iters < monotone_iterations)
                            if not split: break
                            # Find new split
                            argmaxsg = split_gain_tot_flat.argmax()
                            split_feature_sample = torch.div(argmaxsg, max_bin, rounding_mode='floor')
                            split_bin = argmaxsg - split_feature_sample * max_bin
                            split_left = (Xe[split_feature_sample] > split_bin).squeeze()
                            split_right = ~split_left * split_node
                            split_left *= split_node
                            # Compute new leaf weights
                            mu_left = _leaf_prediction_mu_gbdt(gradient[split_left], hessian[split_left], reg_lambda)
                            mu_right = _leaf_prediction_mu_gbdt(gradient[split_right], hessian[split_right], reg_lambda)
                            # Check if new proposed split has a monotonicity constraint
                            split_constraint = monotone_constraints[sample_features[split_feature_sample]]
                            condition = split * (((mu_left < node_min) + (mu_left > node_max) + \
                                                  (mu_right < node_min) + (mu_right > node_max)) + \
                                                 ((split_constraint != 0) * (
                                                             torch.sign(mu_right - mu_left) != split_constraint)))
                            split_iters += 1
                        # Only create a split if there still is a split to make...
                        if split:
                            # Compute min and max values for children nodes
                            if split_constraint == 1:
                                left_node_min = node_min
                                left_node_max = mu_right
                                right_node_min = mu_left
                                right_node_max = node_max
                            elif split_constraint == -1:
                                left_node_min = mu_right
                                left_node_max = node_max
                                right_node_min = node_min
                                right_node_max = mu_left
                            else:
                                left_node_min = node_min
                                left_node_max = node_max
                                right_node_min = node_min
                                right_node_max = node_max
                            # Set left children constraints
                            node_constraints[node_constraint_idx, 0] = 2 * node
                            node_constraints[node_constraint_idx, 1] = torch.maximum(left_node_min, node_constraints[
                                node_constraint_idx, 1])
                            node_constraints[node_constraint_idx, 2] = torch.minimum(left_node_max, node_constraints[
                                node_constraint_idx, 2])
                            node_constraint_idx += 1
                            # Set right children constraints
                            node_constraints[node_constraint_idx, 0] = 2 * node + 1
                            node_constraints[node_constraint_idx, 1] = torch.maximum(right_node_min, node_constraints[
                                node_constraint_idx, 1])
                            node_constraints[node_constraint_idx, 2] = torch.minimum(right_node_max, node_constraints[
                                node_constraint_idx, 2])
                            node_constraint_idx += 1
                            # Create split
                            feature = sample_features[split_feature_sample]
                            nodes_idx[estimator, node_idx] = node
                            nodes_split_feature[estimator, node_idx] = feature
                            nodes_split_bin[estimator, node_idx] = split_bin
                            # Feature importance
                            feature_importance[feature] += split_gain * X_node.shape[1] / n_samples
                            # Assign samples to next node
                            train_nodes += split_left * node + split_right * (node + 1)
                            next_nodes[2 * node_idx] = 2 * node
                            next_nodes[2 * node_idx + 1] = 2 * node + 1
                            node_idx += 1
                        else:
                            leaves_idx, leaves_mu, leaf_idx, yhat_train = \
                                _leaf_prediction_gbdt(gradient[split_node_train], hessian[split_node_train], node,
                                                 estimator, reg_lambda, leaves_idx, leaves_mu,
                                                 leaf_idx, split_node, yhat_train,
                                                 learning_rate)
                    else:
                        # Set left children constraints
                        node_constraints[node_constraint_idx, 0] = 2 * node
                        node_constraint_idx += 1
                        # Set right children constraints
                        node_constraints[node_constraint_idx, 0] = 2 * node + 1
                        node_constraint_idx += 1
                        # Save split information
                        feature = sample_features[split_feature_sample]
                        nodes_idx[estimator, node_idx] = node
                        nodes_split_feature[estimator, node_idx] = feature
                        nodes_split_bin[estimator, node_idx] = split_bin
                        # Feature importance
                        feature_importance[feature] += split_gain * X_node.shape[1] / n_samples
                        # Assign samples to next node
                        train_nodes += split_left * node + split_right * (node + 1)
                        next_nodes[2 * node_idx] = 2 * node
                        next_nodes[2 * node_idx + 1] = 2 * node + 1
                        node_idx += 1
                else:
                    # Save split information
                    feature = sample_features[split_feature_sample]
                    nodes_idx[estimator, node_idx] = node
                    nodes_split_feature[estimator, node_idx] = feature
                    nodes_split_bin[estimator, node_idx] = split_bin
                    # Feature importance
                    feature_importance[feature] += split_gain * X_node.shape[1] / n_samples
                    # Assign samples to next node
                    train_nodes += split_left * node + split_right * (node + 1)
                    next_nodes[2 * node_idx] = 2 * node
                    next_nodes[2 * node_idx + 1] = 2 * node + 1
                    node_idx += 1
            else:
                leaves_idx, leaves_mu, leaf_idx, yhat_train = \
                    _leaf_prediction_gbdt(gradient[split_node_train], hessian[split_node_train], node,
                                     estimator, reg_lambda, leaves_idx, leaves_mu,
                                     leaf_idx, split_node, yhat_train,
                                     learning_rate)
        else:
            leaves_idx, leaves_mu, leaf_idx, yhat_train = \
                _leaf_prediction_gbdt(gradient[split_node_train], hessian[split_node_train], node,
                                 estimator, reg_lambda, leaves_idx, leaves_mu,
                                 leaf_idx, split_node, yhat_train,
                                 learning_rate)

        node = next_nodes[next_node_idx]
        next_node_idx += 1

    return nodes_idx, nodes_split_bin, nodes_split_feature, leaves_idx, \
           leaves_mu, feature_importance, yhat_train
##--##

class GBDT(GBM):
    """
    Gradient Boosting Decision Trees (GBDT).
    """
    def __init__(self, id='GBDT', gbm_para_dict=None, gpu=False, device=None, distributed=False):
        super(GBDT, self).__init__(id=id, gbm_para_dict=gbm_para_dict, gpu=gpu, device=device, distributed=distributed)

    def _init_params(self, params, gpu, device):
        # Set device
        if gpu is True:
           self.torch_device = device
        else:
           print('Training on CPU')
           self.torch_device = torch.device('cpu')
           self.device = 'cpu'



        param_names = ['min_split_gain', 'min_data_in_leaf', 'learning_rate', 'reg_lambda',
                       'max_leaves', 'max_bin', 'n_estimators', 'verbose', 'early_stopping_rounds',
                       'feature_fraction', 'bagging_fraction', 'seed', 'derivatives', 'distribution',
                       'checkpoint', 'tree_correlation', 'monotone_constraints', 'monotone_iterations']
        param_dtypes = ['torch_float', 'torch_float', 'torch_float', 'torch_float',
                        'int', 'int', 'int', 'int', 'int',
                        'torch_float', 'torch_float', 'int', 'str', 'str',
                        'bool', 'torch_float', 'torch_long', 'int']
        param_defaults = [params['min_split_gain'], params['min_data_in_leaf'], params['learning_rate'],
                          params['reg_lambda'],
                          params['max_leaves'], params['max_bin'], params['n_estimators'], 2,
                          params['early_stopping_rounds'],
                          params['feature_fraction'], params['bagging_fraction'], 2147483647, 'exact', 'normal',
                          False, np.log10(self.n_samples) / 100, np.zeros(self.n_features), 1]
        # Initialize all parameters
        for i, param in enumerate(param_names):
            self._init_single_param(param, param_defaults[i], param_dtypes[i], params)
        # Check monotone constraints
        assert self.monotone_constraints.shape[0] == self.n_features, "The number of items in the monotonicity constraint list should be equal to the number of features in your dataset."
        self.any_monotone = torch.any(self.monotone_constraints != 0)

        # Make sure we bound certain parameters
        self.min_data_in_leaf = torch.clamp(self.min_data_in_leaf, 2)
        self.min_split_gain = torch.clamp(self.min_split_gain, 0.0)
        self.feature_samples = (self.feature_fraction * self.n_features).clamp(1, self.n_features).type(torch.int64)
        self.bagging_samples = (self.bagging_fraction * self.n_samples).clamp(1, self.n_samples).type(torch.int64)
        self.monotone_iterations = np.maximum(self.monotone_iterations, 1)
        # Set some additional params
        self.max_nodes = self.max_leaves - 1
        torch.manual_seed(self.seed)  # cpu
        torch.cuda.manual_seed_all(self.seed)
        self.epsilon = 1.0e-4

    def train(self, train_set, objective, metric, valid_set=None, obj_id=None,
              sample_weight=None, eval_sample_weight=None, ranking_obj=False, ranking_metric=False):
        self.train_op(train_set, objective, metric, params=self.gbm_para_dict, valid_set=valid_set,
                      create_tree=_create_tree_gbdt, sample_weight=sample_weight, eval_sample_weight=eval_sample_weight,
                      ranking_obj=ranking_obj, ranking_metric=ranking_metric, probabilistic=False, obj_id=obj_id)

    def save(self, filename):
        """
        Save a PGBM model to a file. The model parameters are saved as numpy arrays and dictionaries.

        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.train(train_set, objective, metric)
            >> model.save('model.pt')

        Args:
            filename (string): name and location to save the model to
        """
        '''
        params = self.params.copy()
        params['learning_rate'] = params['learning_rate'].cpu().numpy()
        #params['tree_correlation'] = params['tree_correlation'].cpu().numpy()
        params['lambda'] = params['lambda'].cpu().numpy()
        params['min_split_gain'] = params['min_split_gain'].cpu().numpy()
        params['min_data_in_leaf'] = params['min_data_in_leaf'].cpu().numpy()

        state_dict = {'nodes_idx': self.nodes_idx[:self.best_iteration].cpu().numpy(),
                      'nodes_split_feature': self.nodes_split_feature[:self.best_iteration].cpu().numpy(),
                      'nodes_split_bin': self.nodes_split_bin[:self.best_iteration].cpu().numpy(),
                      'leaves_idx': self.leaves_idx[:self.best_iteration].cpu().numpy(),
                      'leaves_mu': self.leaves_mu[:self.best_iteration].cpu().numpy(),
                      'feature_importance': self.feature_importance.cpu().numpy(),
                      'best_iteration': self.best_iteration,
                      'params': params,
                      'yhat0': self.yhat_0.cpu().numpy(),
                      'bins': self.bins.cpu().numpy()}
'''
        state_dict = {'nodes_idx': self.nodes_idx[:self.best_iteration].cpu().numpy(),
                      'nodes_split_feature': self.nodes_split_feature[:self.best_iteration].cpu().numpy(),
                      'nodes_split_bin': self.nodes_split_bin[:self.best_iteration].cpu().numpy(),
                      'leaves_idx': self.leaves_idx[:self.best_iteration].cpu().numpy(),
                      'leaves_mu': self.leaves_mu[:self.best_iteration].cpu().numpy(),
                      'feature_importance': self.feature_importance.cpu().numpy(),
                      'best_iteration': self.best_iteration,
                      'bins': self.bins.cpu().numpy(),
                      'min_split_gain': self.min_split_gain.cpu().numpy(),
                      'min_data_in_leaf': self.min_data_in_leaf.cpu().numpy(),
                      'learning_rate': self.learning_rate.cpu().numpy(),
                      'lambda': self.reg_lambda.cpu().numpy(),
                      'max_leaves': self.max_leaves,
                      'max_bin': self.max_bin,
                      'verbose': self.verbose,
                      'early_stopping_rounds': self.early_stopping_rounds,
                      'feature_fraction': self.feature_fraction.cpu().numpy(),
                      'bagging_fraction': self.bagging_fraction.cpu().numpy(),
                      'seed': self.seed,
                      'derivatives': self.derivatives,
                      'distribution': self.distribution,
                      'tree_correlation': self.tree_correlation.cpu().numpy(),
                      }
        with open(filename, 'wb') as handle:
            pickle.dump(state_dict, handle)

    def load(self, filename, device=None):
        """
        Load a PGBM model from a file to a device.

        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.load('model.pt') # Load to default device (cpu)
            >> model.load('model.pt', device=torch.device(0)) # Load to default GPU at index 0

        Args:
            filename (string): location of model file.
            device (torch.device): device to which to load the model. Default = 'cpu'.
        """
        if device is None:
            device = torch.device('cpu')
        with open(filename, 'rb') as handle:
            state_dict = pickle.load(handle)

        torch_float = lambda x: torch.from_numpy(x).float().to(device)
        torch_long = lambda x: torch.from_numpy(x).long().to(device)

        self.nodes_idx = torch_long(state_dict['nodes_idx'])
        self.nodes_split_feature = torch_long(state_dict['nodes_split_feature'])
        self.nodes_split_bin = torch_long(state_dict['nodes_split_bin'])
        self.leaves_idx = torch_long(state_dict['leaves_idx'])
        self.leaves_mu = torch_float(state_dict['leaves_mu'])
        self.feature_importance = torch_float(state_dict['feature_importance'])
        self.best_iteration = state_dict['best_iteration']
        self.params = state_dict['params']
        self.params['learning_rate'] = torch_float(np.array(self.params['learning_rate']))
        #self.params['tree_correlation'] = torch_float(np.array(self.params['tree_correlation']))
        self.params['lambda'] = torch_float(np.array(self.params['lambda']))
        self.params['min_split_gain'] = torch_float(np.array(self.params['min_split_gain']))
        self.params['min_data_in_leaf'] = torch_float(np.array(self.params['min_data_in_leaf']))
        self.yhat_0 = torch_float(np.array(state_dict['yhat0']))
        self.bins = torch_float(state_dict['bins'])  # [num_features, max_bin]
        self.device = device