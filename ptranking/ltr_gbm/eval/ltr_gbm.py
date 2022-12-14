
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
A general framework for evaluating learning-to-rank methods based on the technique of gradient boosting.
"""

import sys
import json
import datetime
import numpy as np

import torch

from ptranking.data.data_utils import SPLIT_TYPE, MSLETOR_SEMI, GBMDataset
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator
from ptranking.ltr_adhoc.eval.parameter import ValidationTape, CVTape, SummaryTape, OptLossTape
from ptranking.ltr_gbm.gbdt_ranker.gbdt_ranker import GBDTRanker, GBDTRankerParameter
from ptranking.ltr_gbm.pgbm_ranker.pgbm_ranker import PGBMRanker, PGBMRankerParameter
from ptranking.ltr_adhoc.eval.parameter import ScoringFunctionParameter
from ptranking.ltr_gbm.eval.gbm_parameter import GBMScoringFunctionParameter

LTR_GBM_MODEL = ['GrowNet', 'GBDTRanker']

class GBMLTREvaluator(LTREvaluator):
    """
    The class for evaluating learning-to-rank methods based on gradient boosting machine
    """
    def __init__(self, frame_id='GBM_LTR', cuda=None):
        super(GBMLTREvaluator, self).__init__(frame_id=frame_id, cuda=cuda)

    def check_consistency(self, data_dict, eval_dict, sf_para_dict):
        """
        Check whether the settings are reasonable in the context of adversarial learning-to-rank
        """
        ''' Part-1: data loading '''
        # TODO-update
        #assert 1 == data_dict['train_batch_size']  # the required setting w.r.t. adversarial LTR

        if data_dict['data_id'] == 'Istella':
            assert eval_dict['do_validation'] is not True  # since there is no validation data

        if data_dict['data_id'] in MSLETOR_SEMI:
            assert data_dict['unknown_as_zero'] is not True  # use original data

        if data_dict['scale_data']:
            scaler_level = data_dict['scaler_level'] if 'scaler_level' in data_dict else None
            assert not scaler_level == 'DATASET'  # not supported setting

        # TODO-update
        assert data_dict['validation_presort']  # Rule of thumb, as validation and test data are for metric-performance
        assert data_dict['test_presort']  # Rule of thumb, as validation and test data are for metric-performance
        #assert 1 == data_dict['validation_batch_size']  # Rule of thumb, as validation and test data are for metric-performance
        #assert 1 == data_dict['test_batch_size']  # Rule of thumb, as validation and test data are for metric-performance

        ''' Part-2: evaluation setting '''
        if eval_dict['mask_label']:  # True is aimed to use supervised data to mimic semi-supervised data by masking
            assert not data_dict['data_id'] in MSLETOR_SEMI

        ''' Part-1: network setting '''

    def load_data(self, eval_dict, data_dict, fold_k, sf_para_dict=None):
        if sf_para_dict['sf_id'] == 'gbdt': # for methods based on GBDT
            file_train, file_vali, file_test = self.determine_files(data_dict, fold_k=fold_k)

            train_data = GBMDataset(split_type=SPLIT_TYPE.Train, file=file_train, data_dict=data_dict,
                                    presort=False, buffer=True).get_data()
            test_data = GBMDataset(split_type=SPLIT_TYPE.Test, file=file_test, data_dict=data_dict,
                                   presort=False, buffer=True).get_data()

            if eval_dict['do_validation'] or eval_dict['do_summary']:  # vali_data is required
                vali_data = GBMDataset(split_type=SPLIT_TYPE.Validation, file=file_vali, data_dict=data_dict,
                                       presort=False, buffer=True).get_data()
            else:
                vali_data = None

            return train_data, test_data, vali_data
        else:
            return super().load_data(eval_dict, data_dict, fold_k)

    def load_ranker(self, sf_para_dict, gb_para_dict):
        """
        Load a ranker correspondingly
        :param sf_para_dict:
        :param model_para_dict:
        :param kwargs:
        :return:
        """
        model_id = gb_para_dict['model_id']

        if model_id in ['GrowNet']:
            ranker = globals()[model_id](sf_para_dict=sf_para_dict, gb_para_dict=gb_para_dict, gpu=self.gpu, device=self.device)
        elif model_id in ['GBDTRanker', 'PGBMRanker']:
            ranker = globals()[model_id](gbm_para_dict=gb_para_dict, gpu=self.gpu, device=self.device)
        else:
            raise NotImplementedError

        return ranker

    def set_scoring_function_setting(self, sf_json=None, debug=None, sf_id=None):
        if sf_json is not None: # since the following module depends on the value of sf_id
            with open(sf_json) as json_file:
                json_dict = json.load(json_file)["SFParameter"]
            sf_id = json_dict['sf_id']

        if sf_id == 'gbdt':
            if sf_json is not None:
                self.sf_parameter = GBMScoringFunctionParameter(sf_json=sf_json)
            else:
                self.sf_parameter = GBMScoringFunctionParameter(debug=debug, sf_id=sf_id)
        else:
            if sf_json is not None:
                self.sf_parameter = ScoringFunctionParameter(sf_json=sf_json)
            else:
                self.sf_parameter = ScoringFunctionParameter(debug=debug, sf_id=sf_id)

    def get_default_scoring_function_setting(self):
        return self.sf_parameter.default_para_dict()

    def iterate_scoring_function_setting(self):
        return self.sf_parameter.grid_search()

    def set_model_setting(self, model_id=None, dir_json=None, debug=False):
        """
        Initialize the parameter class for a specified model
        :param debug:
        :param model_id:
        :return:
        """
        if dir_json is not None:
            para_json = dir_json + model_id + "Parameter.json"
            self.model_parameter = globals()[model_id + "Parameter"](para_json=para_json)
        else: # the 3rd type, where debug-mode enables quick test
            self.model_parameter = globals()[model_id + "Parameter"](debug=debug)

    def gbm_cv_eval_1(self, data_dict=None, eval_dict=None, sf_para_dict=None, gb_para_dict=None):
        """
        Evaluation learning-to-rank methods via k-fold cross validation if there are k folds, otherwise one fold.
        :param data_dict:       settings w.r.t. data
        :param eval_dict:       settings w.r.t. evaluation
        :param sf_para_dict:    settings w.r.t. scoring function
        :param model_para_dict: settings w.r.t. the ltr_adhoc model
        :return:
        """
        '''
        1> early-stopping w.r.t. number of members or trees
        2> better understanding the in-built validation of pgbm
        '''
        self.display_information(data_dict, gb_para_dict)
        #self.check_consistency(data_dict, eval_dict, sf_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, gb_para_dict)

        model_id = gb_para_dict['model_id']
        fold_num, label_type, max_label = data_dict['fold_num'], data_dict['label_type'], data_dict['max_rele_level']
        train_presort, validation_presort, test_presort = \
            data_dict['train_presort'], data_dict['validation_presort'], data_dict['test_presort']
        # for quick access of common evaluation settings
        epochs, loss_guided = eval_dict['epochs'], eval_dict['loss_guided']
        vali_k, log_step, cutoffs   = eval_dict['vali_k'], eval_dict['log_step'], eval_dict['cutoffs']
        do_vali, vali_metric, do_summary = eval_dict['do_validation'], eval_dict['vali_metric'], eval_dict['do_summary']

        sf_para_dict[sf_para_dict['sf_id']].update(dict(num_features=data_dict['num_features']))
        gbm = self.load_ranker(gb_para_dict=gb_para_dict, sf_para_dict=sf_para_dict)

        cv_tape = CVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=do_vali)
        for fold_k in range(1, fold_num + 1):   # evaluation over k-fold data
            train_data, test_data, vali_data = self.load_data(eval_dict, data_dict, fold_k)

            gbm.start_over(train_data=train_data, max_rele_level=data_dict['max_rele_level'])  # reset with the same random initialization

            if do_vali:
                vali_tape = ValidationTape(fold_k=fold_k, num_epochs=epochs, validation_metric=vali_metric,
                                           validation_at_k=vali_k, dir_run=self.dir_run)
            if do_summary:
                summary_tape = SummaryTape(do_validation=do_vali, cutoffs=cutoffs, label_type=label_type,
                                           train_presort=train_presort, test_presort=test_presort, gpu=self.gpu)
            if not do_vali and loss_guided:
                opt_loss_tape = OptLossTape(gpu=self.gpu)

            for member_id in range(1, gbm.num_members + 1):
                epoch_k = member_id # TODO TBA

                member_ranker = gbm.ini_new_member(member_id=member_id)
                gbm.train_member_ranker(member_ranker=member_ranker, train_data=train_data)
                gbm.add_member(member_ranker)

                if member_id > 3 and gbm.CT:
                    gbm.corrective_train(train_data=train_data)

                ranker = gbm # TODO TBA
                torch_fold_k_epoch_k_loss = None # TODO TBA

                if (do_summary or do_vali) and (epoch_k % log_step == 0 or epoch_k == 1):  # stepwise check
                    if do_vali:     # per-step validation score
                        torch_vali_metric_value = ranker.validation(vali_data=vali_data, k=vali_k, device='cpu',
                                                                    vali_metric=vali_metric, label_type=label_type,
                                                                    max_label=max_label, presort=validation_presort)
                        vali_metric_value = torch_vali_metric_value.squeeze(-1).data.numpy()
                        vali_tape.epoch_validation(ranker=ranker, epoch_k=epoch_k, metric_value=vali_metric_value)

                    if do_summary:  # summarize per-step performance w.r.t. train, test
                        summary_tape.epoch_summary(ranker=ranker, torch_epoch_k_loss=torch_fold_k_epoch_k_loss,
                                                   train_data=train_data, test_data=test_data,
                                                   vali_metric_value=vali_metric_value if do_vali else None)

                elif loss_guided:  # stopping check via epoch-loss
                    early_stopping = opt_loss_tape.epoch_cmp_loss(fold_k=fold_k, epoch_k=epoch_k,
                                                                  torch_epoch_k_loss=torch_fold_k_epoch_k_loss)
                    if early_stopping: break

            if do_summary:  # track
                summary_tape.fold_summary(fold_k=fold_k, dir_run=self.dir_run, train_data_length=train_data.__len__())

            if do_vali: # using the fold-wise optimal model for later testing based on validation data
                ranker.load(vali_tape.get_optimal_path())
                vali_tape.clear_fold_buffer(fold_k=fold_k)
            else: # buffer the model after a fixed number of training-epoches if no validation is deployed
                fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/',
                            name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')

            cv_tape.fold_evaluation(model_id=model_id, ranker=ranker, test_data=test_data, max_label=max_label, fold_k=fold_k)

        ndcg_cv_avg_scores = cv_tape.get_cv_performance()
        return ndcg_cv_avg_scores

    def setup_eval(self, data_dict, eval_dict, sf_para_dict, model_para_dict):
        """
        Finalize the evaluation setting correspondingly
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        self.dir_run  = self.setup_output(data_dict, eval_dict)
        if eval_dict['do_log'] and not self.eval_setting.debug:
            time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
            sys.stdout = open(self.dir_run + '_'.join(['log', time_str]) + '.txt', "w")

    def gbm_cv_eval(self, data_dict=None, eval_dict=None, sf_para_dict=None, gb_para_dict=None):
        """
        Evaluation learning-to-rank methods via k-fold cross validation if there are k folds, otherwise one fold.
        :param data_dict:       settings w.r.t. data
        :param eval_dict:       settings w.r.t. evaluation
        :param sf_para_dict:    settings w.r.t. scoring function
        :param model_para_dict: settings w.r.t. the ltr_adhoc model
        :return:
        """
        # TODO-update
        '''
        1> early-stopping w.r.t. number of members or trees
        2> better understanding the in-built validation of pgbm
        '''
        self.display_information(data_dict, gb_para_dict)
        #self.check_consistency(data_dict, eval_dict, sf_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, gb_para_dict)

        model_id = gb_para_dict['model_id']
        fold_num, label_type, max_label = data_dict['fold_num'], data_dict['label_type'], data_dict['max_rele_level']
        # TODO-update
        train_presort, validation_presort, test_presort = \
            data_dict['train_presort'], data_dict['validation_presort'], data_dict['test_presort']
        # for quick access of common evaluation settings
        epochs, loss_guided = eval_dict['epochs'], eval_dict['loss_guided']
        vali_k, log_step, cutoffs   = eval_dict['vali_k'], eval_dict['log_step'], eval_dict['cutoffs']
        do_vali, vali_metric, do_summary = eval_dict['do_validation'], eval_dict['vali_metric'], eval_dict['do_summary']

        if not sf_para_dict['sf_id'] in ['gbdt']:
            sf_para_dict[sf_para_dict['sf_id']].update(dict(num_features=data_dict['num_features']))

        gbm_ranker = self.load_ranker(gb_para_dict=gb_para_dict, sf_para_dict=sf_para_dict)

        cv_tape = CVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=do_vali)
        for fold_k in range(1, fold_num + 1):   # evaluation over k-fold data
            train_data, test_data, vali_data = self.load_data(eval_dict=eval_dict, data_dict=data_dict, fold_k=fold_k,
                                                              sf_para_dict=sf_para_dict)
            gbm_ranker.full_train(train_data, vali_data)

            cv_tape.fold_evaluation(model_id=model_id, ranker=gbm_ranker, test_data=test_data, max_label=max_label, fold_k=fold_k)

        ndcg_cv_avg_scores = cv_tape.get_cv_performance()
        return ndcg_cv_avg_scores

    def grid_run(self, model_id=None, sf_id=None, dir_json=None, debug=False, data_id=None, dir_data=None, dir_output=None):
        """
        Explore the effects of different hyper-parameters of a model based on grid-search
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """
        if dir_json is not None:
            data_eval_sf_json = dir_json + 'Data_Eval_ScoringFunction.json'
            self.set_data_setting(data_json=data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=data_eval_sf_json)
            self.set_eval_setting(debug=debug, eval_json=data_eval_sf_json)
            self.set_model_setting(model_id=model_id, dir_json=dir_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
            self.set_model_setting(debug=debug, model_id=model_id)

        self.declare_global(model_id=model_id)

        ''' select the best setting through grid search '''
        vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]
        max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
        k_index = cutoffs.index(vali_k)
        max_common_para_dict, max_sf_para_dict, max_model_para_dict = None, None, None

        for data_dict in self.iterate_data_setting():
            for eval_dict in self.iterate_eval_setting():
                assert self.eval_setting.check_consistence(vali_k=vali_k, cutoffs=cutoffs) # a necessary consistence

                for sf_para_dict in self.iterate_scoring_function_setting():
                    for model_para_dict in self.iterate_model_setting():
                        curr_cv_avg_scores = self.gbm_cv_eval(data_dict=data_dict, eval_dict=eval_dict,
                                                              sf_para_dict=sf_para_dict, gb_para_dict=model_para_dict)
                        if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                            max_cv_avg_scores, max_sf_para_dict, max_eval_dict, max_model_para_dict = \
                                                           curr_cv_avg_scores, sf_para_dict, eval_dict, model_para_dict

        # log max setting
        self.log_max(data_dict=data_dict, eval_dict=max_eval_dict,
                     max_cv_avg_scores=max_cv_avg_scores, sf_para_dict=max_sf_para_dict,
                     log_para_str=self.model_parameter.to_para_string(log=True, given_para_dict=max_model_para_dict))


    def point_run(self, debug=False, model_id=None, data_id=None, dir_data=None, dir_output=None, sf_id=None, reproduce=False):
        self.set_eval_setting(debug=debug, dir_output=dir_output)
        self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
        data_dict = self.get_default_data_setting()
        eval_dict = self.get_default_eval_setting()

        self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
        sf_para_dict = self.get_default_scoring_function_setting()

        self.set_model_setting(debug=debug, model_id=model_id)
        gb_para_dict = self.get_default_model_setting()

        self.gbm_cv_eval(data_dict=data_dict, eval_dict=eval_dict, gb_para_dict=gb_para_dict, sf_para_dict=sf_para_dict)

    def run(self, debug=False, model_id=None, sf_id=None, config_with_json=None, dir_json=None,
            data_id=None, dir_data=None, dir_output=None, grid_search=False, reproduce=False):

        if config_with_json:
            assert dir_json is not None
            if reproduce:
                self.point_run(debug=debug, model_id=model_id, dir_json=dir_json, reproduce=reproduce)
            else:
                self.grid_run(debug=debug, model_id=model_id, dir_json=dir_json)
        else:
            assert sf_id in ['pointsf', 'listsf', 'gbdt']
            if grid_search:
                self.grid_run(debug=debug, model_id=model_id, sf_id=sf_id,
                              data_id=data_id, dir_data=dir_data, dir_output=dir_output)
            else:
                self.point_run(debug=debug, model_id=model_id, sf_id=sf_id,
                               data_id=data_id, dir_data=dir_data, dir_output=dir_output, reproduce=reproduce)

