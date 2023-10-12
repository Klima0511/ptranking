#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""
import os

from torch import nn

from ptranking.ltr_gandalf.gandalf.base.GFLU import Add
from ptranking.ltr_gandalf.gandalf.base.gandalf import GANDALFBackbone

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from itertools import product
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.metric_utils import get_delta_ndcg
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from ptranking.ltr_node.node.base.nn_utils import entmax15, entmoid15





class gandalf(NeuralRanker):
    def __init__(self, sf_para_dict =None,model_para_dict=None, gpu=False, device=None, **kwargs):
        super(gandalf, self).__init__(id='gandalf', sf_para_dict=sf_para_dict,
                                     gpu=gpu, device=device)
        self.model_para_dict = model_para_dict
        self.device = device
        self.gpu = gpu
        self.sigma = model_para_dict['sigma']
        self.lr = model_para_dict['lr']
        self.opt = model_para_dict['opt']
        self.weight_decay = model_para_dict['weight']


        # GANDALF Backbone
        self.backbone = GANDALFBackbone(
            cat_embedding_dims = [],
            n_continuous_features=model_para_dict['input_dim'],
            gflu_stages=model_para_dict['gflu_stages'],
            gflu_dropout=model_para_dict['gflu_dropout'],
            gflu_feature_init_sparsity=model_para_dict['gflu_feature_init_sparsity'],
            learnable_sparsity=True,
            batch_norm_continuous_input=True,
            embedding_dropout=0
        )

        # Embedding Layer
        self.embedding_layer = self.backbone._build_embedding_layer()
        self.final_layer = nn.Linear(model_para_dict['input_dim'], 1)

        # GANDALF Head
        self.T0 = nn.Parameter(torch.rand(model_para_dict['input_dim']), requires_grad=True)
        self.head = nn.Sequential(self.embedding_layer, self.backbone, Add(self.T0))

        # Final Model
        self.model = nn.Sequential(
            self.head,
            self.final_layer
        ).to(self.device)

        # Assuming you have a method named config_optimizer in NeuralRanker
        self.config_optimizer()

    # ... rest of your class methods ...




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
        input_dict = {"continuous": batch_q_doc_vectors}#模型要求

        _batch_preds = self.model(input_dict)

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

        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        X = batch_q_doc_vectors.view(-1, num_features)
        input_dict = {"continuous": X}
        # compute model output
        scores = self.model(input_dict)

        """
        if isinstance(scores, list):
            scores = [x.cpu().detach().numpy() for x in scores]
        else:
            scores = scores.cpu().detach().numpy()
        """

        batch_preds = scores.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds


    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
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

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.model.state_dict(), dir + name)

    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.model.load_state_dict(torch.load(file_model, map_location=device))


###### Parameter of gandalf ######

class gandalfParameter(ModelParameter):
    ''' Parameter class for gandalf '''

    def __init__(self, debug=False, para_json=None):
        super(gandalfParameter, self).__init__(model_id='gandalf', para_json=para_json)
        self.gandalf_para_dict = None
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for gandalf
        :return:
        """
        self.gandalf_para_dict = dict(model_id=self.model_id,
                                     embedding_dims=0,
                                     gflu_stages=4,
                                     gflu_dropout=6,  # sensitive
                                     gflu_feature_init_sparsity=entmax15,
                                     lr = 0.002,
                                     opt ="Adam",
                                     sigma = 1.0,
                                     weight=1e-3
                                     )
        return self.gandalf_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        gandalf_para_dict = given_para_dict if given_para_dict is not None else self.gandalf_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        embedding_dims,  gflu_stages, gflu_dropout, gflu_feature_init_sparsity, sigma,lr,opt,weight = \
            gandalf_para_dict['embedding_dims'],  gandalf_para_dict['gflu_stages'], \
            gandalf_para_dict['gflu_dropout'], gandalf_para_dict['gflu_feature_init_sparsity'], \
            gandalf_para_dict['sigma'],gandalf_para_dict['lr'],\
            gandalf_para_dict['weight'],gandalf_para_dict['opt']

        para_string = s2.join([s1.join(['embedding_dims', str(embedding_dims)]),
                               s1.join(['gflu_stages', str(gflu_stages)]), s1.join(['gflu_dropout', str(gflu_dropout)]),
                               s1.join(['gflu_feature_init_sparsity', str(gflu_feature_init_sparsity)]),
                               s1.join(['sigma', str(sigma)]),s1.join(['lr', str(lr)]),s1.join(['weight_decay', str(weight)]),
                               s1.join(['opt', str(opt)]),
                               ])

        return para_string

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaRank
        """
        if self.use_json:
            choice_embedding_dims = self.json_dict['embedding_dims']
            choice_gflu_stages = self.json_dict['gflu_stages']
            choice_gflu_dropout = self.json_dict['gflu_dropout']
            choice_gflu_feature_init_sparsity = self.json_dict['gflu_feature_init_sparsity']
            choice_sigma = self.json_dict['sigma']
            choice_lr = self.json_dict['lr']
            choice_opt = self.json_dict['opt']
            choice_weight_decay = self.json_dict['weight']
        else:
            choice_embedding_dims = 0
            choice_gflu_stages = 2
            choice_gflu_dropout = 0.0
            choice_gflu_feature_init_sparsity = 0.0
            choice_sigma = 1.0
            choice_lr = 0.01
            choice_opt = "Adam"
            choice_weight_decay = 1e-3
        for embedding_dims, gflu_stages, gflu_dropout, gflu_feature_init_sparsity,  sigma,lr,opt,weight in product(
                choice_embedding_dims,
                choice_gflu_stages,
                choice_gflu_dropout,
                choice_gflu_feature_init_sparsity,
                choice_sigma,
                choice_lr,
                choice_opt,
                choice_weight_decay):
            self.gandalf_para_dict = dict(model_id=self.model_id,
                                         embedding_dims = embedding_dims,
                                         gflu_stages = gflu_stages,
                                         gflu_dropout = gflu_dropout,
                                         gflu_feature_init_sparsity = gflu_feature_init_sparsity,
                                         lr=lr,
                                         opt=opt,
                                         sigma=sigma,
                                         weight = weight,
                                         )

            yield self.gandalf_para_dict
