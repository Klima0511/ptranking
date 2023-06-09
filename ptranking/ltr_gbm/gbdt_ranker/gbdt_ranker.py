
import torch

from itertools import product

from ptranking.ltr_gbm.base.gbdt.gbdt import GBDT
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.metric.adhoc.adhoc_metric import torch_ndcg_at_ks, torch_nerr_at_ks, torch_ap_at_ks,\
    torch_precision_at_ks
from ptranking.ltr_gbm.gbdt_ranker.gbdt_ranker_util import lambdarank_objective, nDCG_metric
from ptranking.ltr_gbm.pgbm_ranker.pgbm_ranker_util import mseloss_objective

class GBDTRanker(GBDT):
    """ A group of learning-to-rank models based on GBDT """
    def __init__(self, id='GBDTRanker', gbm_para_dict=None, gpu=False, device=None, distributed=False):
        super(GBDT, self).__init__(id=id, gbm_para_dict=gbm_para_dict, gpu=gpu, device=device, distributed=distributed)

    def get_custom_obj(self, obj_id):
        if obj_id == 'mse':
            return mseloss_objective
        elif obj_id == 'listnet':
            return NotImplementedError
        elif obj_id == 'lambdarank':
            return lambdarank_objective
        else:
            raise NotImplementedError

    def get_objective_settings(self):
        obj_id = self.gbm_para_dict['obj']
        objective = self.get_custom_obj(obj_id=obj_id)
        if obj_id in ['lambdarank']:
            metric, ranking_obj, ranking_metric = nDCG_metric, True, True
        elif obj_id in ['mse']:
            metric, ranking_obj, ranking_metric = nDCG_metric, True, True
        else:
            raise NotImplementedError

        return objective, metric, ranking_obj, ranking_metric

    def adhoc_performance_at_ks(self, test_data=None, ks=[1, 5, 10], label_type=LABEL_TYPE.MultiLabel, max_label=None,
                                presort=False, device='cpu', need_per_q=False):
        X_test, y_test, group_test = test_data

        y_point_preds = self.gbdt_ranker.predict(X_test)
        y_std = self.gbdt_ranker._convert_array(y_test)

        num_queries = 0
        sum_ndcg_at_ks = torch.zeros(len(ks))
        sum_nerr_at_ks = torch.zeros(len(ks))
        sum_ap_at_ks = torch.zeros(len(ks))
        sum_p_at_ks = torch.zeros(len(ks))

        if need_per_q: list_per_q_p, list_per_q_ap, list_per_q_nerr, list_per_q_ndcg = [], [], [], []

        #for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
        head = 0
        for i in range(group_test.shape[0]):
            gr = int(group_test[i])
            per_query_std_labels = y_std[head:head + gr]
            per_query_preds = y_point_preds[head:head + gr]
            head += gr

            _, per_query_sorted_inds = torch.sort(per_query_preds, descending=True)
            _batch_predict_rankings = per_query_std_labels[per_query_sorted_inds]
            batch_predict_rankings = _batch_predict_rankings.view(1, -1)
            _batch_ideal_rankings, _ = torch.sort(per_query_std_labels, descending=True)
            batch_ideal_rankings = _batch_ideal_rankings.view(1, -1)
            batch_predict_rankings = batch_predict_rankings.cpu()
            batch_ideal_rankings = batch_ideal_rankings.cpu()

            batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings,
                                                ks=ks, label_type=label_type, device=device)
            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))

            batch_nerr_at_ks = torch_nerr_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings, max_label=max_label,
                                                ks=ks, label_type=label_type, device=device)
            sum_nerr_at_ks = torch.add(sum_nerr_at_ks, torch.sum(batch_nerr_at_ks, dim=0))

            batch_ap_at_ks = torch_ap_at_ks(batch_predict_rankings=batch_predict_rankings,
                                            batch_ideal_rankings=batch_ideal_rankings, ks=ks, device=device)
            sum_ap_at_ks = torch.add(sum_ap_at_ks, torch.sum(batch_ap_at_ks, dim=0))

            batch_p_at_ks = torch_precision_at_ks(batch_predict_rankings=batch_predict_rankings, ks=ks, device=device)
            sum_p_at_ks = torch.add(sum_p_at_ks, torch.sum(batch_p_at_ks, dim=0))

            if need_per_q:
                list_per_q_p.append(batch_p_at_ks)
                list_per_q_ap.append(batch_ap_at_ks)
                list_per_q_nerr.append(batch_nerr_at_ks)
                list_per_q_ndcg.append(batch_ndcg_at_ks)

            num_queries += 1

        avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
        avg_nerr_at_ks = sum_nerr_at_ks / num_queries
        avg_ap_at_ks = sum_ap_at_ks / num_queries
        avg_p_at_ks = sum_p_at_ks / num_queries

        if need_per_q:
            return avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks, \
                   list_per_q_ndcg, list_per_q_nerr, list_per_q_ap, list_per_q_p
        else:
            return avg_ndcg_at_ks, avg_nerr_at_ks, avg_ap_at_ks, avg_p_at_ks


    def full_train(self, train_data, vali_data):
        self.gbdt_ranker = GBDT(gbm_para_dict=self.gbm_para_dict, gpu=self.gpu, device=self.device, distributed=False)

        objective, metric, ranking_obj, ranking_metric = self.get_objective_settings()

        self.gbdt_ranker.train(train_data, objective=objective, metric=metric, valid_set=vali_data,
                               ranking_obj=ranking_obj, ranking_metric=ranking_metric, obj_id=self.gbm_para_dict['obj'])

###### Parameter of GBDTRanker ######

class GBDTRankerParameter(ModelParameter):
    ''' Parameter class for GBDTRanker '''
    def __init__(self, debug=False, para_json=None):
        super(GBDTRankerParameter, self).__init__(model_id='GBDTRanker', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        self.gbm_para_dict = dict(model_id=self.model_id,
                                  obj='lambdarank', learning_rate =0.01, max_bin =256, max_leaves =32,
                                  min_split_gain = 0.0, min_data_in_leaf = 2, reg_lambda = 1.0,
                                  n_estimators = 5, feature_fraction = 1.0, bagging_fraction = 1.0,
                                  early_stopping_rounds = 100, eval_at = 5)
        return self.gbm_para_dict

    def grid_search(self):
        """
        Iterator of parameter settings for GBDTRanker
        """
        custom_dict = dict()
        if self.use_json:
            choice_obj = self.json_dict['obj']
            choice_lr = self.json_dict['learning_rate']
            choice_max_bin = self.json_dict['max_bin']
            choice_max_leaves = self.json_dict['max_leaves']
            choice_min_split_gain = self.json_dict['min_split_gain']
            choice_min_data_in_leaf = self.json_dict['min_data_in_leaf']
            choice_reg_lambda = self.json_dict['reg_lambda']
            choice_n_estimators = self.json_dict['n_estimators']
            choice_feature_fraction = self.json_dict['feature_fraction']
            choice_bagging_fraction = self.json_dict['bagging_fraction']
            choice_early_stopping_rounds = self.json_dict['early_stopping_rounds']
            eval_at = self.json_dict['eval_at']
        else:
            # common setting when using in-built lightgbm's ranker
            choice_obj = ['lambdarank'] if self.debug else ['lambdarank']
            choice_lr = [0.05, 0.01] if self.debug else [0.05, 0.01]
            choice_max_bin = [256] if self.debug else [256]
            choice_max_leaves = [32] if self.debug else [32]
            choice_min_split_gain = [0.0] if self.debug else [0.0]
            choice_min_data_in_leaf = [2] if self.debug else [2]
            choice_reg_lambda = [1.0] if self.debug else [1.0]
            choice_n_estimators = [5] if self.debug else [5]
            choice_feature_fraction = [1.0] if self.debug else [1.0]
            choice_bagging_fraction = [1.0] if self.debug else [1.0]
            choice_early_stopping_rounds = [100] if self.debug else [100]
            eval_at = 5

        for obj, learning_rate, max_bin, max_leaves, min_split_gain,\
            min_data_in_leaf, reg_lambda, n_estimators, feature_fraction,\
            bagging_fraction, early_stopping_rounds in \
                product(choice_obj, choice_lr, choice_max_bin, choice_max_leaves, choice_min_split_gain,
                        choice_min_data_in_leaf, choice_reg_lambda, choice_n_estimators, choice_feature_fraction,
                        choice_bagging_fraction, choice_early_stopping_rounds):
            self.gbm_para_dict = {'model_id': self.model_id,
                                'obj': obj,
                                'learning_rate': learning_rate,
                                'max_bin': max_bin,
                                'max_leaves': max_leaves,
                                'min_split_gain': min_split_gain,
                                'min_data_in_leaf': min_data_in_leaf,
                                'reg_lambda': reg_lambda,
                                'n_estimators': n_estimators,
                                'feature_fraction': feature_fraction,
                                'bagging_fraction': bagging_fraction,
                                'early_stopping_rounds': early_stopping_rounds,
                                'eval_at': eval_at, # validation cutoff
                                'verbosity': -1}
            yield self.gbm_para_dict