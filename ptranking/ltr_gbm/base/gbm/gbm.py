
import numpy as np

from pathlib import Path

import torch
from torch.autograd import grad
from torch.utils.cpp_extension import load


## Structure clarification ##
# the current implementation is based on the 2022.11.29 - version of pgbm: https://github.com/elephaint/pgbm
"""
Gradient boosting machines (GBMs) is the parent-level class object, which will be extended as
(1) traditional gradient boosting decision tree (GBDT);
(2) probabilistic gradient boosting machines (PGBM);

The common components between GBDT and PGBM are as follows:
_create_X_splits
_create_feature_bins

The differences between GBDT and PGBM are as follows:
<1> from the view of theoretical comparison:
for both a tree and a forest, GBDT's output is a deterministic estimation, 
PGBM's output is a distribution (e.g., mu and variance in the case of gaussian estimation).
<2> from the view of implementation:
the implementation of the following functions are customized:
_create_tree
_leaf_prediction
_predict_tree
_predict_forest
train
predict
"""

#%% Load custom kernel
current_path = Path(__file__).parent.absolute()
if torch.cuda.is_available():
    load(name="split_decision",
        sources=[f'{current_path}/splitgain_cuda.cpp',
                 f'{current_path}/splitgain_kernel.cu'],
        is_python_module=False,
        verbose=True)
else:
    load(name="split_decision",
        sources=[f'{current_path}/splitgain_cpu.cpp'],
        is_python_module=False,
        verbose=True)

##- Commonly required torch.jit.script -##
@torch.jit.script
def _create_X_splits(X: torch.Tensor, bins: torch.Tensor):
    # Pre-compute split decisions for X
    max_bin = bins.shape[1]
    dtype_split = torch.uint8 if max_bin <= 256 else torch.int16
    X_splits = torch.zeros((X.shape[1], X.shape[0]), device=X.device, dtype=dtype_split)
    for i in range(max_bin):
        # bins[:, i] corresponds to a vector [number_features, 1], i.e., the i-th bin for all features
        X_splits += (X > bins[:, i]).T
    # the cumulative value of X_splits is meaningful, which indicates the bin to be used.
    return X_splits

@torch.jit.script
def _create_feature_bins(X: torch.Tensor, max_bin: int = 256):
    # Create array that contains the bins
    bins = torch.zeros((X.shape[1], max_bin), device=X.device)
    quantiles = torch.linspace(0, 1, max_bin, device=X.device)
    # For each feature, create max_bins based on frequency bins.
    for i in range(X.shape[1]):
        xs = X[:, i]
        # II
        '''
        Computes the q-th quantiles of each row of the input tensor along the dimension dim.
        If q is a 1D tensor, the first dimension of the output represents the quantiles and has size equal to the size of q
        :[num_samples] (w.r.t. one feature), [num_quantiles] -> [num_quantiles, 1]
        Q: requiring sorted=True for torch.unique()? A: not needed due to quantile() operation with given 0.1, 0.2, etc.
        '''
        current_bin = torch.unique(torch.quantile(xs, quantiles))
        # A bit inefficiency created here... some features usually have less than max_bin values (e.g. 1/0 indicator features).
        bins[i, :current_bin.shape[0]] = current_bin
        bins[i, current_bin.shape[0]:] = current_bin.max()

    return bins

@torch.jit.script
def _predict_forest_mu(X: torch.Tensor, nodes_idx: torch.Tensor, nodes_split_bin: torch.Tensor,
                    nodes_split_feature: torch.Tensor, leaves_idx: torch.Tensor,
                    leaves_mu: torch.Tensor, lr: torch.Tensor,
                    best_iteration: int):
    # Parallel prediction of a tree ensemble - mean only
    nodes_predict = torch.ones((X.shape[1], best_iteration), device=X.device, dtype=torch.int64)
    unique_nodes = torch.unique(nodes_idx, sorted=True)
    unique_nodes = unique_nodes[unique_nodes != 0]
    node = torch.ones(1, device = X.device, dtype=torch.int64)
    mu = torch.zeros((X.shape[1], best_iteration), device=X.device, dtype=torch.float32)
    # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
    leaf_idx = torch.eq(node, leaves_idx)
    index = torch.any(leaf_idx, dim=1)
    mu[:, index] = leaves_mu[leaf_idx]
    # Loop over nodes
    for node in unique_nodes:
        # Select current node information
        split_node = nodes_predict == node
        node_idx = (nodes_idx == node)
        current_features = (nodes_split_feature * node_idx).sum(1)
        current_bins = (nodes_split_bin * node_idx).sum(1, keepdim=True)
        # Split node
        split_left = (X[current_features] > current_bins).T
        split_right = ~split_left * split_node
        split_left *= split_node
        # Check if children are leaves
        leaf_idx_left = torch.eq(2 * node, leaves_idx)
        leaf_idx_right = torch.eq(2 * node + 1, leaves_idx)
        # Update mu and variance with left leaf prediction
        mu += split_left * (leaves_mu * leaf_idx_left).sum(1)
        sum_left = leaf_idx_left.sum(1)
        nodes_predict += (1 - sum_left) * split_left * node
        # Update mu and variance with right leaf prediction
        mu += split_right * (leaves_mu * leaf_idx_right).sum(1)
        sum_right = leaf_idx_right.sum(1)
        nodes_predict += (1  - sum_right) * split_right * (node + 1)

    # Each prediction only for the amount of learning rate in the ensemble
    mu = -lr * mu.sum(1)

    return mu

@torch.jit.script
def _predict_tree_mu(X: torch.Tensor, nodes_idx: torch.Tensor,
                     nodes_split_bin: torch.Tensor, nodes_split_feature: torch.Tensor,
                     leaves_idx: torch.Tensor, leaves_mu: torch.Tensor,
                     lr: torch.Tensor):
    # Get prediction for a single tree
    nodes_predict = torch.ones(X.shape[1], device=X.device, dtype=torch.int)
    mu = torch.zeros(X.shape[1], device=X.device, dtype=torch.float32)
    node = torch.ones(1, device=X.device, dtype=torch.int64)
    # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
    leaf_idx = torch.eq(node, leaves_idx)
    mu += (leaves_mu * leaf_idx).sum()
    # Loop over nodes
    for node in nodes_idx:
        if node == 0: break
        # Select current node information
        split_node = nodes_predict == node
        node_idx = (nodes_idx == node)
        current_feature = (nodes_split_feature * node_idx).sum()
        current_bin = (nodes_split_bin * node_idx).sum()
        # Split node
        split_left = (X[current_feature] > current_bin).squeeze()
        split_right = ~split_left * split_node
        split_left *= split_node
        # Check if children are leaves
        leaf_idx_left = torch.eq(2 * node, leaves_idx)
        leaf_idx_right = torch.eq(2 * node + 1, leaves_idx)
        # Left leaf prediction
        mu += split_left * (leaves_mu * leaf_idx_left).sum()
        sum_left = leaf_idx_left.sum()
        nodes_predict += (1 - sum_left) * split_left * node
        # Right leaf prediction
        mu += split_right * (leaves_mu * leaf_idx_right).sum()
        sum_right = leaf_idx_right.sum()
        nodes_predict += (1 - sum_right) * split_right * (node + 1)

    return -lr * mu
##--##

class GBM():
    '''
    A general framework for gradient boosting machines (GBMs)
    '''
    def __init__(self, id='GBMs', gbm_para_dict=None, gpu=False, device=None, distributed=False):
        super(GBM, self).__init__()
        self.id = id
        self.cwd = Path().cwd()
        self.new_model = True
        self.gbm_para_dict = gbm_para_dict

        self.gpu = gpu
        self.device = device
        self.distributed = distributed

    def _init_params(self, gbm_para_dict):
        pass

    # copy from reference project
    def _init_single_param(self, param_name, default, dtype, params=None):
        # Lambda function to convert parameter to correct dtype
        if dtype == 'int' or dtype == 'str' or dtype == 'bool' or dtype == 'other':
            convert = lambda x: x
        elif dtype == 'torch_float':
            convert = lambda x: torch.tensor(x, dtype=torch.float32, device=self.torch_device)
        elif dtype == 'torch_long':
            convert = lambda x: torch.tensor(x, dtype=torch.int64, device=self.torch_device)
        # Check if the parameter is in the supplied dict, else use existing or default
        if param_name in params:
            setattr(self, param_name, convert(params[param_name]))
        else:
            if not hasattr(self, param_name):
                setattr(self, param_name, convert(default))

    def create_feature_bins(self, X, max_bin):
        bins = _create_feature_bins(X, max_bin)
        return bins

    # copy from reference project
    def _objective_approx(self, yhat_train, y, levels=None):
        yhat = yhat_train.detach()
        yhat.requires_grad = True
        yhat_upper = yhat + self.epsilon
        yhat_lower = yhat - self.epsilon
        loss = self.loss(yhat, y, levels)
        loss_upper = self.loss(yhat_upper, y, levels)
        loss_lower = self.loss(yhat_lower, y, levels)
        gradient = grad(loss, yhat)[0]
        gradient_upper = grad(loss_upper, yhat_upper)[0]
        gradient_lower = grad(loss_lower, yhat_lower)[0]
        hessian = (gradient_upper - gradient_lower) / (2 * self.epsilon)

        return gradient, hessian

    # copy from reference project
    def _convert_array(self, array):
        if type(array) == np.ndarray:
            array = torch.from_numpy(array).float()
        elif type(array) == torch.Tensor:
            array = array.float()

        return array.to(self.device)

    def create_X_splits(self, X, bins):
        X_splits = _create_X_splits(X, bins)
        return X_splits

    # by customizing train() from reference project, where {} are added {ranking_obj=False, ranking_metric=False, probabilistic=False};
    def train_op(self, train_set, objective, metric, params=None, valid_set=None, create_tree=None,
                 sample_weight=None, eval_sample_weight=None,
                 ranking_obj=False, ranking_metric=False, probabilistic=False, obj_id=None):
        # the customization is based on: ranking_obj=False, ranking_metric=False, probabilistic=False
        # Create parameters
        best_score = 0
        self.n_samples = train_set[0].shape[0]
        self.n_features = train_set[0].shape[1]
        if params == None:
            params = {}
        self._init_params(params,self.gpu,self.device)
        # Create train data
        X_train, y_train = self._convert_array(train_set[0]), self._convert_array(train_set[1]).squeeze()
        # Set objective & metric
        if self.derivatives == 'exact':
            self.objective = objective
        else:
            self.loss = objective
            self.objective = self._objective_approx
        self.metric = metric
        # Initialize predictions
        n_samples = torch.tensor(X_train.shape[0], device=X_train.device)
        y_train_sum = y_train.sum()
        # Pre-allocate arrays
        nodes_idx = torch.zeros((self.n_estimators, self.max_nodes), dtype=torch.int64, device=self.torch_device)
        nodes_split_feature = torch.zeros_like(nodes_idx)
        nodes_split_bin = torch.zeros_like(nodes_idx)
        leaves_idx = torch.zeros((self.n_estimators, self.max_leaves), dtype=torch.int64, device=self.torch_device)
        leaves_mu = torch.zeros((self.n_estimators, self.max_leaves), dtype=torch.float32, device=self.torch_device)
        if probabilistic: leaves_var = torch.zeros_like(leaves_mu)
        # Continue training from existing model or train new model, depending on whether a model was loaded.
        if self.new_model:
            self.initial_estimate = y_train_sum / n_samples
            self.best_iteration = 0
            yhat_train = self.initial_estimate.repeat(n_samples)
            # [num_features, max_bin]
            self.bins = self.create_feature_bins(X_train, self.max_bin)
            self.feature_importance = torch.zeros(self.n_features, dtype=torch.float32, device=self.torch_device)
            self.nodes_idx = nodes_idx
            self.nodes_split_feature = nodes_split_feature
            self.nodes_split_bin = nodes_split_bin
            self.leaves_idx = leaves_idx
            self.leaves_mu = leaves_mu
            if probabilistic: self.leaves_var = leaves_var
            start_iteration = 0
        else:
            yhat_train = self.predict(X_train)
            self.nodes_idx = torch.cat((self.nodes_idx, nodes_idx))
            self.nodes_split_feature = torch.cat((self.nodes_split_feature, nodes_split_feature))
            self.nodes_split_bin = torch.cat((self.nodes_split_bin, nodes_split_bin))
            self.leaves_idx = torch.cat((self.leaves_idx, leaves_idx))
            self.leaves_mu = torch.cat((self.leaves_mu, leaves_mu))
            if probabilistic: self.leaves_var = torch.cat((self.leaves_var, leaves_var))
            start_iteration = self.best_iteration
        # Initialize
        train_nodes = torch.ones(self.n_samples, dtype=torch.int64, device=self.torch_device)
        # Pre-compute split decisions for X_train
        X_train_splits = self.create_X_splits(X_train, self.bins)
        # Prepare logs for train metrics
        self.train_metrics = torch.zeros(self.n_estimators, dtype=torch.float32, device=self.torch_device)
        # Initialize validation
        validate = False
        #self.best_score = torch.tensor(0., device=self.torch_device, dtype=torch.float32)
        if valid_set is not None:#Has valid_data
            validate = True
            early_stopping = 0
            X_validate, y_validate = self._convert_array(valid_set[0]), self._convert_array(valid_set[1]).squeeze()
            # Prepare logs for validation metrics
            self.validation_metrics = torch.zeros(self.n_estimators, dtype=torch.float32, device=self.torch_device)
            if self.new_model:
                yhat_validate = self.initial_estimate.repeat(y_validate.shape[0])
                #self.best_score += float('inf')
            else:
                yhat_validate = self.predict(X_validate)
                if ranking_metric:
                    validation_metric = metric(yhat_validate, y_validate, eval_sample_weight, group=valid_set[2])
                else:
                    validation_metric = metric(yhat_validate, y_validate, eval_sample_weight)
                #self.best_score += validation_metric

            # Pre-compute split decisions for X_validate
            X_validate_splits = self.create_X_splits(X_validate, self.bins)

        # Retrieve initial loss and gradient
        if ranking_obj and not obj_id=='mse':
            gradient, hessian = self.objective(yhat_train, y_train, sample_weight,self.device, group=train_set[2])
        else:
            gradient, hessian = self.objective(yhat_train, y_train, sample_weight)
        # Loop over estimators
        for estimator in range(start_iteration, self.n_estimators + start_iteration):
            # Retrieve bagging batch
            samples = ~torch.round(
                torch.rand(self.n_samples, device=self.torch_device) * (1 / (2 * self.bagging_fraction))).bool()
            sample_features = torch.arange(self.n_features,
                                           device=self.torch_device) if self.feature_fraction == 1.0 else torch.randperm(
                self.n_features, device=self.torch_device)[:self.feature_samples]
            # Create tree
            if probabilistic:
                self.nodes_idx, self.nodes_split_bin, self.nodes_split_feature, self.leaves_idx, \
                self.leaves_mu, self.leaves_var, self.feature_importance, yhat_train = \
                    create_tree(X_train_splits, gradient,
                                 hessian, estimator, train_nodes,
                                 self.nodes_idx, self.nodes_split_bin, self.nodes_split_feature,
                                 self.leaves_idx, self.leaves_mu, self.leaves_var,
                                 self.feature_importance, yhat_train, self.learning_rate,
                                 self.max_nodes, samples, sample_features, self.max_bin,
                                 self.min_data_in_leaf, self.reg_lambda,
                                 self.min_split_gain, self.any_monotone,
                                 self.monotone_constraints, self.monotone_iterations)
            else:
                self.nodes_idx, self.nodes_split_bin, self.nodes_split_feature, self.leaves_idx, \
                self.leaves_mu, self.feature_importance, yhat_train = \
                    create_tree(X_train_splits, gradient,
                                hessian, estimator, train_nodes,
                                self.nodes_idx, self.nodes_split_bin, self.nodes_split_feature,
                                self.leaves_idx, self.leaves_mu,
                                self.feature_importance, yhat_train, self.learning_rate,
                                self.max_nodes, samples, sample_features, self.max_bin,
                                self.min_data_in_leaf, self.reg_lambda,
                                self.min_split_gain, self.any_monotone,
                                self.monotone_constraints, self.monotone_iterations)
            # Compute new gradient and hessian
            if ranking_obj and not obj_id=='mse':
                gradient, hessian = self.objective(yhat_train, y_train, sample_weight, self.device, group=train_set[2])
            else:
                gradient, hessian = self.objective(yhat_train, y_train, sample_weight)
            # Compute metric
            if ranking_metric:
                train_metric = self.metric(yhat_train, y_train, sample_weight, self.device, group=train_set[2])
            else:
                train_metric = self.metric(yhat_train, y_train, sample_weight)

            self.train_metrics[estimator] = train_metric
            # Reset train nodes
            train_nodes.fill_(1)
            # Validation statistics
            if validate:
                yhat_validate += _predict_tree_mu(X_validate_splits, self.nodes_idx[estimator],
                                                  self.nodes_split_bin[estimator],
                                                  self.nodes_split_feature[estimator],
                                                  self.leaves_idx[estimator], self.leaves_mu[estimator],
                                                  self.learning_rate)

                if ranking_metric:
                    validation_metric = self.metric(yhat_validate, y_validate, eval_sample_weight, self.device,group=valid_set[2])
                else:
                    validation_metric = self.metric(yhat_validate, y_validate, eval_sample_weight)

                self.validation_metrics[estimator] = validation_metric
                if self.verbose > 1:
                    print(f"Estimator {estimator}/{self.n_estimators + start_iteration}, Train metric: {train_metric:.4f}, Validation metric: {validation_metric:.4f}")
                #TODO consistent setting
                if validation_metric > best_score:#?????
                    best_score = validation_metric
                    self.best_iteration = estimator
                    #self.save(f'{"/data/tan_haonan/Output/MSLR-WEB30K/gpu_grid_PGBMRanker/PGBMRanker_SF__MSLRWEB30K_MiD_10_MiR_1_TrBat_1_TrPresort_EP_300_V_AP@5_QS_StandardScaler"}\checkpoint{self.best_iteration}')
                    early_stopping = 1
                else:
                    early_stopping += 1
                    if early_stopping == self.early_stopping_rounds:
                        break
            else:
                if self.verbose > 1:
                    print(f"Estimator {estimator}/{self.n_estimators + start_iteration}, Train metric: {train_metric:.4f}")
                self.best_iteration = estimator + 1

            # Save current model checkpoint to current working directory
        #self.save(f'{"/data/tan_haonan/Output/MSLR-WEB30K/gpu_grid_GBDTRanker/"}\checkpoint{self.best_iteration}')

        # Truncate tree arrays
        self.nodes_idx = self.nodes_idx[:self.best_iteration]
        self.nodes_split_bin = self.nodes_split_bin[:self.best_iteration]
        self.nodes_split_feature = self.nodes_split_feature[:self.best_iteration]
        self.leaves_idx = self.leaves_idx[:self.best_iteration]
        self.leaves_mu = self.leaves_mu[:self.best_iteration]
        if probabilistic: self.leaves_var = self.leaves_var[:self.best_iteration]
        self.train_metrics = self.train_metrics[:self.best_iteration]
        if validate:
            self.validation_metrics = self.validation_metrics[:self.best_iteration]

    # copy from reference project within PGBM
    def predict(self, X, parallel=True):
        """
        Generate point estimates/forecasts for a given sample set X
        """
        X = self._convert_array(X)
        initial_estimate = self.initial_estimate.repeat(X.shape[0])
        mu = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        # Construct split decision tensor
        X_test_splits = self.create_X_splits(X, self.bins)

        # Predict samples
        if parallel:
            mu = _predict_forest_mu(X_test_splits, self.nodes_idx,
                                    self.nodes_split_bin, self.nodes_split_feature,
                                    self.leaves_idx, self.leaves_mu,
                                    self.learning_rate, self.best_iteration)
        else:
            for estimator in range(self.best_iteration):
                mu += _predict_tree_mu(X_test_splits, self.nodes_idx[estimator],
                                       self.nodes_split_bin[estimator], self.nodes_split_feature[estimator],
                                       self.leaves_idx[estimator], self.leaves_mu[estimator],
                                       self.learning_rate)

        return initial_estimate + mu

    def save(self, filename):
        """
        Save a GBM model to a file. The model parameters are saved as numpy arrays and dictionaries.
        """
        pass

    def load(self, filename, device=None):
        """
        Load a GBM model from a file to a device.
        """
        pass
