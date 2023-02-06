#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""
import os

from ptranking.ltr_adhoc.pointwise.rank_mse import rankMSE_loss_function

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from scipy.sparse import csc_matrix
import numpy as np
from itertools import product

import scipy
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.metric_utils import get_delta_ndcg
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from ptranking.ltr_ntree.tabnet.base import tab_network

from ptranking.ltr_global import ltr_seed


class TabNet(NeuralRanker):
    '''

    '''

    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(TabNet, self).__init__(id='TabNet', sf_para_dict=sf_para_dict,
                                     gpu=gpu, device=device)
        self.model_para_dict = model_para_dict
        self.augmentations = None
        self.sigma = model_para_dict['sigma']
        self.lr = model_para_dict['lr']
        self.opt = model_para_dict['opt']
        self.weight_decay = model_para_dict['weight']
        self.lambda_sparse = model_para_dict['lambda_sparse']
    def init(self):  # initialize tab_network with model_para_dict
        """Setup the network and explain matrix."""
        torch.manual_seed(ltr_seed)
        """
        self.network = tab_network.TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
        ).to(self.device)
        """
        self.network = tab_network.TabNet(
            input_dim=self.model_para_dict['input_dim'],
            output_dim=1,
            n_d=self.model_para_dict['n_d'],
            n_a=self.model_para_dict['n_d'],
            n_steps=self.model_para_dict['n_steps'],
            gamma=self.model_para_dict['gamma'],
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=1,
            n_independent=self.model_para_dict['n_independent'],
            n_shared=self.model_para_dict['n_shared'],
            epsilon=self.model_para_dict['epsilon'],
            virtual_batch_size=self.model_para_dict['virtual_batch_size'],
            momentum=self.model_para_dict['momentum'],
            mask_type=self.model_para_dict['mask_type'],
        ).to(self.device)

        self.reducing_matrix = self.create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

        self.config_optimizer()

    def get_parameters(self):
        '''
        Get the trainable parameters of the scoring function.
        '''
        return self.network.parameters()

    def _compute_feature_importances(self, loader):
        """Compute global feature importance.

        Parameters
        ----------
        loader : `torch.utils.data.Dataloader`
            Pytorch dataloader.

        """
        feature_importances_ = np.zeros((self.network.post_embed_dim))
        for batch_ids, batch_q_doc_vectors, batch_std_labels in loader:
            batch_q_doc_vectors = batch_q_doc_vectors.to(self.device).float()
            a1 = batch_q_doc_vectors.shape[0]
            a2 = batch_q_doc_vectors.shape[1]
            a4 = batch_q_doc_vectors.shape[2]
            a3 = a1 * a2
            a = batch_q_doc_vectors.reshape(a3, a4)
            M_explain, masks = self.network.forward_masks(a)
            feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()

        feature_importances_ = csc_matrix.dot(
            feature_importances_, self.reducing_matrix
        )
        feature_importances_ = feature_importances_ / np.sum(feature_importances_)
        return feature_importances_

    def config_optimizer(self):
        '''
        Configure the optimizer correspondingly.
        '''
        if 'Adam' == self.opt:
            self.optimizer = optim.Adam(self.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif 'RMS' == self.opt:
            self.optimizer = optim.RMSprop(self.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif 'Adagrad' == self.opt:
            self.optimizer = optim.Adagrad(self.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)

    def create_explain_matrix(self, input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
        """
        This is a computational trick.
        In order to rapidly sum importances from same embeddings
        to the initial index.

        Parameters
        ----------
        input_dim : int
            Initial input dim
        cat_emb_dim : int or list of int
            if int : size of embedding for all categorical feature
            if list of int : size of embedding for each categorical feature
        cat_idxs : list of int
            Initial position of categorical features
        post_embed_dim : int
            Post embedding inputs dimension

        Returns
        -------
        reducing_matrix : np.array
            Matrix of dim (post_embed_dim, input_dim)  to performe reduce
        """

        if isinstance(cat_emb_dim, int):
            all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
        else:
            all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

        acc_emb = 0
        nb_emb = 0
        indices_trick = []
        for i in range(input_dim):
            if i not in cat_idxs:
                indices_trick.append([i + acc_emb])
            else:
                indices_trick.append(
                    range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)
                )
                acc_emb += all_emb_impact[nb_emb]
                nb_emb += 1

        reducing_matrix = np.zeros((post_embed_dim, input_dim))
        for i, cols in enumerate(indices_trick):
            reducing_matrix[cols, i] = 1

        return scipy.sparse.csc_matrix(reducing_matrix)

    def eval_mode(self):
        self.network.eval()

    def train_mode(self):
        self.network.train()

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        query_number, num_docs, num_features = batch_q_doc_vectors.size()#12x8x46
        # print("batch_size, num_docs, num_features", batch_size, num_docs, num_features)

        ###tabnet from _train_batch
        X = batch_q_doc_vectors.view(-1, num_features).to(self.device).float()#96x46
        if self.augmentations is not None:
            X, y = self.augmentations(X, )

        for param in self.network.parameters():
            param.grad = None

        output, M_loss = self.network(X)

        # print("output", output.size())
        batch_preds = output.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]

        """
        loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        loss = loss - self.lambda_sparse * M_loss

        # Perform backward pass and optimization
        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()
        """
        ###tabnet

        # _batch_preds = self.point_sf(batch_q_doc_vectors)
        # batch_preds = _batch_preds.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        # return batch_preds
        return batch_preds,M_loss

    def predict(self, batch_q_doc_vectors):
        '''
        The relevance prediction.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        # batch_preds = self.forward(batch_q_doc_vectors)
        # return batch_preds

        ### tabnet from _predict_epoch
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        X = batch_q_doc_vectors.view(-1, num_features).to(self.device).float()
        # compute model output
        scores, _ = self.network(X)

        """
        if isinstance(scores, list):
            scores = [x.cpu().detach().numpy() for x in scores]
        else:
            scores = scores.cpu().detach().numpy()
        """

        batch_preds = scores.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds
        ### tabnet

    def train_op(self, batch_q_doc_vectors, batch_std_labels, **kwargs):
        '''
        The training operation over a batch of queries.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance labels for documents associated with the same query.
        @param kwargs: optional arguments
        @return:
        '''
        stop_training = False
        batch_preds,M_loss = self.forward(batch_q_doc_vectors)

        if 'epoch_k' in kwargs and kwargs['epoch_k'] % self.stop_check_freq == 0:
            stop_training = self.stop_training(batch_preds)

        return self.custom_loss_function(batch_preds, batch_std_labels,M_loss, **kwargs), stop_training
    def custom_loss_function(self, batch_preds, batch_std_labels,M_loss, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''


        assert 'label_type' in kwargs and LABEL_TYPE.MultiLabel == kwargs['label_type']
        label_type = kwargs['label_type']
        assert 'presort' in kwargs and kwargs['presort'] is True  # aiming for direct usage of ideal ranking

        # sort documents according to the predicted relevance
        batch_descending_preds, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
        # reorder batch_stds correspondingly so as to make it consistent.
        # BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor
        batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)

        batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=batch_descending_preds,
                                                             batch_std_labels=batch_predict_rankings,
                                                             sigma=self.sigma)

        batch_delta_ndcg = get_delta_ndcg(batch_ideal_rankings=batch_std_labels,
                                          batch_predict_rankings=batch_predict_rankings,
                                          label_type=label_type, device=self.device)

        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1),
                                             weight=torch.triu(batch_delta_ndcg, diagonal=1), reduction='none')

        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        batch_loss = batch_loss - self.lambda_sparse * M_loss
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss





    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.network.state_dict(), dir + name)

    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.network.load_state_dict(torch.load(file_model, map_location=device))


###### Parameter of TabNet ######

class TabNetParameter(ModelParameter):
    ''' Parameter class for TabNet '''

    def __init__(self, debug=False, para_json=None):
        super(TabNetParameter, self).__init__(model_id='TabNet', para_json=para_json)
        self.tabnet_para_dict = None
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for TabNet
        :return:
        """
        self.tabnet_para_dict = dict(model_id=self.model_id,
                                     n_d=8,
                                     n_a=8,
                                     n_steps=4,
                                     gamma=0.003,  # sensitive
                                     n_independent=4,
                                     n_shared=4,
                                     epsilon=1e-5,
                                     mask_type="entmax",  # sparsemax | entmax
                                      virtual_batch_size = 32,
                                       momentum = 0.02,
                                        lr = 0.002,
                                        opt ="Adam",
                                        sigma = 1.0,
                                     weight=1e-3,
                                     lambda_sparse = 1e-3
                                     )
        return self.tabnet_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        tabnet_para_dict = given_para_dict if given_para_dict is not None else self.tabnet_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        n_d, n_a, n_steps, gamma, n_independent, n_shared, epsilon, mask_type,virtual_batch_size,momentum,sigma,lr,weight,opt,lambda_sparse = \
            tabnet_para_dict['n_d'], tabnet_para_dict['n_d'], tabnet_para_dict['n_steps'], \
            tabnet_para_dict['gamma'], tabnet_para_dict['n_independent'], \
            tabnet_para_dict['n_shared'], tabnet_para_dict['epsilon'], \
            tabnet_para_dict['mask_type'],tabnet_para_dict['virtual_batch_size'],\
            tabnet_para_dict['momentum'],tabnet_para_dict['sigma'],tabnet_para_dict['lr'],\
            tabnet_para_dict['weight'],tabnet_para_dict['opt'],tabnet_para_dict['lambda_sparse']

        para_string = s2.join([s1.join(['n_d', str(n_d)]), s1.join(['n_a', str(n_a)]),
                               s1.join(['n_steps', str(n_steps)]), s1.join(['gamma', str(gamma)]),
                               s1.join(['n_independent', str(n_independent)]),
                               s1.join(['n_shared', str(n_shared)]),
                               s1.join(['epsilon', str(epsilon)]), s1.join(['mask_type', str(mask_type)]),
                               s1.join(['virtual_batch_size', str(virtual_batch_size)]),s1.join(['momentum', str(momentum)]),
                               s1.join(['sigma', str(sigma)]),s1.join(['lr', str(lr)]),s1.join(['weight_decay', str(weight)]),
                               s1.join(['opt', str(opt)]),s1.join(['lambda_sparse',str(lambda_sparse)])
                               ])

        return para_string

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaRank
        """
        if self.use_json:
            choice_n_d = self.json_dict['n_d/n_a']
            choice_n_steps = self.json_dict['n_steps']
            choice_gamma = self.json_dict['gamma']
            choice_n_independent = self.json_dict['n_independent']
            choice_n_shared = self.json_dict['n_shared']
            choice_epsilon = self.json_dict['epsilon']
            choice_mask_type = self.json_dict['mask_type']
            choice_sigma = self.json_dict['sigma']
            choice_virtual_batch_size = self.json_dict['virtual_batch_size']
            choice_momentum = self.json_dict['momentum']
            choice_lr = self.json_dict['lr']
            choice_opt = self.json_dict['opt']
            choice_weight_decay = self.json_dict['weight']
            choice_lambda_sparse = self.json_dict['lambda_sparse']
        else:
            choice_n_d = 8
            choice_n_steps = 3
            choice_gamma = 0.03
            choice_n_independent = 4
            choice_n_shared = 4
            choice_epsilon = 1e-5
            choice_mask_type = "sparsemax"
            choice_sigma = 1.0
            choice_virtual_batch_size = 24
            choice_momentum = 0.02
            choice_lr = 0.02
            choice_opt = "Adam"
            choice_weight_decay = 1e-3
            choice_lambda_sparse = 1e-3
        for n_d, n_steps, gamma, n_independent, n_shared, epsilon, sigma, mask_type, virtual_batch_size, momentum,lr,opt,weight,lambda_sparse in product(
                choice_n_d,
                choice_n_steps,
                choice_gamma,
                choice_n_independent,
                choice_n_shared,
                choice_epsilon,
                choice_sigma,
                choice_mask_type,
                choice_virtual_batch_size,
                choice_momentum,
                choice_lr,
                choice_opt,
                choice_weight_decay,
                choice_lambda_sparse):
            self.tabnet_para_dict = dict(model_id=self.model_id,
                                         n_d=n_d,
                                         n_steps=n_steps,
                                         gamma=gamma,  # sensitive
                                         n_independent=n_independent,
                                         n_shared=n_shared,
                                         epsilon=epsilon,
                                         mask_type=mask_type,# sparsemax | entmax
                                         virtual_batch_size=virtual_batch_size,
                                         momentum=momentum,
                                         lr=lr,
                                         opt=opt,
                                         sigma=sigma,
                                         weight = weight,
                                         lambda_sparse = lambda_sparse
                                         )

            yield self.tabnet_para_dict
