#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""
import os

from torch import nn

from ptranking.ltr_adhoc.pointwise.rank_mse import rankMSE_loss_function

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from scipy.sparse import csc_matrix
import numpy as np
from itertools import product
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from ptranking.ltr_node.node.base.arch import DenseBlock

from ptranking.ltr_node.node.base.utils import get_latest_file, iterate_minibatches, check_numpy, process_in_chunks
from ptranking.ltr_node.node.base.nn_utils import to_one_hot, entmax15, entmoid15, Lambda
from ptranking.ltr_global import ltr_seed


class node(NeuralRanker):
    '''

    '''

    def __init__(self,  sf_para_dict =None,model_para_dict=None, gpu=False, device=None, **kwargs):
        super(node, self).__init__(id='node', sf_para_dict=sf_para_dict,
                                      gpu=gpu, device=device)
        tesorlist = []
        self.model_para_dict = model_para_dict
        self.device = device
        self.gpu = gpu
        self.sigma = model_para_dict['sigma']
        self.lr = model_para_dict['lr']
        self.opt = model_para_dict['opt']
        self.weight_decay = model_para_dict['weight']
        input_dim = self.model_para_dict['input_dim']
        layer_dim = self.model_para_dict['layer_dim']
        num_layers = self.model_para_dict['num_layers']
        tree_dim = self.model_para_dict['tree_dim']
        depth = self.model_para_dict['depth']
        choice_function = self.model_para_dict['choice_function']
        bin_function = self.model_para_dict['bin_function']
        self.stop_check_freq = 1




        self.model = nn.Sequential(DenseBlock(input_dim, layer_dim, num_layers=num_layers, tree_dim=tree_dim, depth=depth, flatten_output=False,
                                                  choice_function=entmax15, bin_function=entmoid15),
                                   Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
                                   ).to(self.device)
        self.config_optimizer()
    def init(self):
        for tensor in list(self.model.parameters()):
            del tensor




    def get_parameters(self):
        '''
        Get the trainable parameters of the scoring function.
        '''
        return self.model.parameters()


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


    def eval_mode(self):
        self.model.eval()

    def train_mode(self):
        self.model.train()

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        batch_q_doc_vectors = batch_q_doc_vectors.reshape(-1, num_features)

        _batch_preds = self.model(batch_q_doc_vectors)
        batch_preds = _batch_preds.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds

    def predict(self, batch_q_doc_vectors):
        '''
        The relevance prediction.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        # batch_preds = self.forward(batch_q_doc_vectors)
        # return batch_preds

        ### node from _predict_epoch
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        X = batch_q_doc_vectors.view(-1, num_features)
        # compute model output
        scores = self.model(X)



        """
        if isinstance(scores, list):
            scores = [x.cpu().detach().numpy() for x in scores]
        else:
            scores = scores.cpu().detach().numpy()
        """

        batch_preds = scores.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds
        ### node


    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:

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


        self.optimizer.zero_grad()
        batch_loss.backward(retain_graph = False)
        print(torch.cuda.memory_allocated('cuda:0') / 1024 ** 2, 'MB')  # 显示已分配的显存量
        print(torch.cuda.memory_cached('cuda:0') / 1024 ** 2, 'MB')  # 显示缓存的显存量

        self.optimizer.step()
         '''
        batch_loss = rankMSE_loss_function(batch_preds, batch_std_labels)

        self.optimizer.zero_grad()
        a=torch.cuda.memory_allocated('cuda:1')/ 1024 ** 2
        batch_loss.backward()
        b=torch.cuda.memory_allocated('cuda:1')/ 1024 ** 2
        # 显示缓存的显存量
        self.optimizer.step()
        np_batch_loss = batch_loss.detach().cpu().numpy()


        return np_batch_loss





    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.model.state_dict(), dir + name)

    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.model.load_state_dict(torch.load(file_model, map_location=device))


###### Parameter of node ######

class nodeParameter(ModelParameter):
    ''' Parameter class for node '''

    def __init__(self, debug=False, para_json=None):
        super(nodeParameter, self).__init__(model_id='node', para_json=para_json)
        self.node_para_dict = None
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for node
        :return:
        """
        self.node_para_dict = dict(model_id=self.model_id,
                                     layer_dim=1024,
                                     num_layers=2,
                                     tree_dim=4,
                                     depth=6,  # sensitive
                                     choice_function=entmax15,
                                     bin_function=entmoid15,
                                     lr = 0.002,
                                     opt ="Adam",
                                     sigma = 1.0,
                                     weight=1e-3
                                     )
        return self.node_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        node_para_dict = given_para_dict if given_para_dict is not None else self.node_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        layer_dim, num_layers, tree_dim, depth, choice_function,bin_function,  sigma,lr,opt,weight = \
            node_para_dict['layer_dim'], node_para_dict['num_layers'], node_para_dict['tree_dim'], \
            node_para_dict['depth'], node_para_dict['choice_function'], \
            node_para_dict['bin_function'],node_para_dict['sigma'],node_para_dict['lr'],\
            node_para_dict['weight'],node_para_dict['opt']

        para_string = s2.join([s1.join(['layer_dim', str(layer_dim)]), s1.join(['num_layers', str(num_layers)]),
                               s1.join(['tree_dim', str(tree_dim)]), s1.join(['depth', str(depth)]),
                               s1.join(['choice_function', str(choice_function)]),
                               s1.join(['bin_function', str(bin_function)]),
                               s1.join(['sigma', str(sigma)]),s1.join(['lr', str(lr)]),s1.join(['weight_decay', str(weight)]),
                               s1.join(['opt', str(opt)]),
                               ])

        return para_string

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaRank
        """
        if self.use_json:
            choice_layer_dim = self.json_dict['layer_dim']
            choice_num_layers = self.json_dict['num_layers']
            choice_tree_dim = self.json_dict['tree_dim']
            choice_depth = self.json_dict['depth']
            choice_choice_function = self.json_dict['choice_function']
            choice_bin_function = self.json_dict['bin_function']
            choice_sigma = self.json_dict['sigma']
            choice_lr = self.json_dict['lr']
            choice_opt = self.json_dict['opt']
            choice_weight_decay = self.json_dict['weight']
        else:
            choice_layer_dim = 8
            choice_num_layers = 3
            choice_tree_dim = 0.03
            choice_depth = 4
            choice_choice_function = entmax15
            choice_bin_function = "entmoid15"
            choice_sigma = 1.0
            choice_lr = 0.02
            choice_opt = "Adam"
            choice_weight_decay = 1e-3
        for layer_dim, num_layers, tree_dim, depth, choice_function,bin_function,  sigma,lr,opt,weight in product(
                choice_layer_dim,
                choice_num_layers,
                choice_tree_dim,
                choice_depth,
                choice_choice_function,
                choice_bin_function,
                choice_sigma,
                choice_lr,
                choice_opt,
                choice_weight_decay):
            self.node_para_dict = dict(model_id=self.model_id,
                                         layer_dim = layer_dim,
                                         num_layers = num_layers,
                                         tree_dim = tree_dim,
                                         depth = depth,
                                         choice_function = choice_function,
                                         bin_function = bin_function,
                                         lr=lr,
                                         opt=opt,
                                         sigma=sigma,
                                         weight = weight,
                                         )

            yield self.node_para_dict
