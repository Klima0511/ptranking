#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import datetime
from matplotlib import pyplot as plt
import numpy as np
from ptranking.metric.metric_utils import metric_results_to_string
from ptranking.base.ranker import LTRFRAME_TYPE
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator
from ptranking.data.data_utils import MSLETOR_SEMI, MSLETOR_LIST
from ptranking.ltr_adhoc.eval.parameter import ValidationTape, CVTape, SummaryTape, OptLossTape
from ptranking.ltr_ntree.tabnet.tabnet import TabNet,TabNetParameter
LTR_NeuralTree_MODEL = ['TabNet']


class NeuralTreeLTREvaluator(LTREvaluator):
    """
    The class for evaluating different neural-tree-based learning to rank methods.
    """
    def __init__(self, frame_id=LTRFRAME_TYPE.Probabilistic, cuda=None):
        super(NeuralTreeLTREvaluator, self).__init__(frame_id=frame_id, cuda=cuda)

    def check_consistency(self, data_dict, eval_dict):
        """
        Check whether the settings are reasonable in the context of adhoc learning-to-rank
        """
        ''' Part-1: data loading '''

        if data_dict['data_id'] == 'Istella':
            assert eval_dict['do_validation'] is not True  # since there is no validation data

        if data_dict['data_id'] in MSLETOR_SEMI:
            assert data_dict['train_presort'] is not True  # due to the non-labeled documents
            if data_dict['binary_rele']:  # for unsupervised dataset, it is required for binarization due to '-1' labels
                assert data_dict['unknown_as_zero']
        else:
            assert data_dict['unknown_as_zero'] is not True  # since there is no non-labeled documents

        if data_dict['data_id'] in MSLETOR_LIST:  # for which the standard ltr_adhoc of each query is unique
            assert 1 == data_dict['train_batch_size']

        if data_dict['scale_data']:
            scaler_level = data_dict['scaler_level'] if 'scaler_level' in data_dict else None
            assert not scaler_level == 'DATASET'  # not supported setting

        assert data_dict['validation_presort']  # Rule of thumb setting for adhoc learning-to-rank
        assert data_dict['test_presort']  # Rule of thumb setting for adhoc learning-to-rank

        ''' Part-2: evaluation setting '''

        if eval_dict['mask_label']:  # True is aimed to use supervised data to mimic semi-supervised data by masking
            assert not data_dict['data_id'] in MSLETOR_SEMI

    def setup_output(self, data_dict=None, eval_dict=None):
        """
        Update output directory
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        model_id = self.model_parameter.model_id
        grid_search, do_vali, dir_output = eval_dict['grid_search'], eval_dict['do_validation'], eval_dict['dir_output']
        mask_label = eval_dict['mask_label']

        if grid_search:
            dir_root = dir_output + '_'.join(['gpu', 'grid', model_id]) + '/' if self.gpu else dir_output + '_'.join(
                ['grid', model_id]) + '/'
        else:
            dir_root = dir_output

        eval_dict['dir_root'] = dir_root
        if not os.path.exists(dir_root): os.makedirs(dir_root)

        # sf_str = self.sf_parameter.to_para_string()
        data_eval_str = '_'.join([self.data_setting.to_data_setting_string(),
                                  self.eval_setting.to_eval_setting_string()])
        if mask_label:
            data_eval_str = '_'.join([data_eval_str, 'MaskLabel', 'Ratio', '{:,g}'.format(eval_dict['mask_ratio'])])

        # file_prefix = '_'.join([model_id, 'SF', sf_str, data_eval_str])
        file_prefix = '_'.join([model_id, data_eval_str])

        if data_dict['scale_data']:
            if data_dict['scaler_level'] == 'QUERY':
                file_prefix = '_'.join([file_prefix, 'QS', data_dict['scaler_id']])
            else:
                file_prefix = '_'.join([file_prefix, 'DS', data_dict['scaler_id']])

        dir_run = dir_root + file_prefix + '/'  # run-specific outputs

        model_para_string = self.model_parameter.to_para_string()
        if len(model_para_string) > 0:
            dir_run = dir_run + model_para_string + '/'

        eval_dict['dir_run'] = dir_run
        if not os.path.exists(dir_run):
            os.makedirs(dir_run)

        return dir_run

    def setup_eval(self, data_dict, eval_dict, model_para_dict):
        """
        Finalize the evaluation setting correspondingly
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        # num_features=data_dict['num_features']
        self.dir_run = self.setup_output(data_dict, eval_dict)

        if eval_dict['do_log'] and not self.eval_setting.debug:
            time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
            sys.stdout = open(self.dir_run + '_'.join(['log', time_str]) + '.txt', "w")

        # if self.do_summary: self.summary_writer = SummaryWriter(self.dir_run + 'summary')
        if not model_para_dict['model_id'] in ['MDPRank', 'ExpectedUtility', 'WassRank']:
            """
            Aiming for efficient batch processing, please use a large batch_size, e.g., {train_rough_batch_size, validation_rough_batch_size, test_rough_batch_size = 300, 300, 300}
            """
            # assert data_dict['train_rough_batch_size'] > 1

    def set_model_setting(self, model_id=None, dir_json=None, debug=False):
        """
        Initialize the parameter class for a specified model
        :param debug:
        :param model_id:
        :return:
        """
        if model_id in ['TabNet']:
            if dir_json is not None:
                para_json = dir_json + model_id + "Parameter.json"
                self.model_parameter = globals()[model_id + "Parameter"](para_json=para_json)
            else:  # the 3rd type, where debug-mode enables quick test
                self.model_parameter = globals()[model_id + "Parameter"](debug=debug)
        else:
            raise NotImplementedError

    def load_ranker(self, model_para_dict):
        """
        Load a ranker correspondingly
        :param sf_para_dict:
        :param model_para_dict:
        :param kwargs:
        :return:
        """
        model_id = model_para_dict['model_id']

        if model_id in ['TabNet']:
            sf_para_dict = dict(sf_id=None, opt='Adam', lr=None)
            ranker = globals()[model_id](sf_para_dict=sf_para_dict, model_para_dict=model_para_dict, gpu=self.gpu,
                                         device=self.device)
        else:
            raise NotImplementedError

        return ranker
    def kfold_cv_reproduce(self, data_dict=None, eval_dict=None, model_para_dict=None):
        self.display_information(data_dict, model_para_dict)
        self.check_consistency(data_dict, eval_dict)
        model_para_dict.update(dict(input_dim=data_dict['num_features']))
        model_id = model_para_dict['model_id']
        fold_num, max_label = data_dict['fold_num'], data_dict['max_rele_level']
        cutoffs, do_vali = eval_dict['cutoffs'], eval_dict['do_validation']
        cv_tape = CVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=do_vali, reproduce=True)
        ranker = self.load_ranker(model_para_dict=model_para_dict)

        model_exp_dir = self.setup_output(data_dict, eval_dict)
        for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
            ranker.init()  # initialize or reset with the same random initialization

            _, test_data, _ = self.load_data(eval_dict, data_dict, fold_k)

            cv_tape.fold_evaluation_reproduce(ranker=ranker, test_data=test_data, dir_run=model_exp_dir,
                                              max_label=max_label, fold_k=fold_k, model_id=model_id, device=self.device)

        ndcg_cv_avg_scores = cv_tape.get_cv_performance()
        return ndcg_cv_avg_scores
    def kfold_cv_eval(self, data_dict=None, eval_dict=None, model_para_dict=None):
        """
        Evaluation learning-to-rank methods via k-fold cross validation if there are k folds, otherwise one fold.
        :param data_dict:       settings w.r.t. data
        :param eval_dict:       settings w.r.t. evaluation
        :param sf_para_dict:    settings w.r.t. scoring function
        :param model_para_dict: settings w.r.t. the ltr_adhoc model
        :return:
        """
        self.display_information(data_dict, model_para_dict)
        self.check_consistency(data_dict, eval_dict)

        # setting
        model_para_dict.update(dict(input_dim=data_dict['num_features']))

        ranker = self.load_ranker(model_para_dict=model_para_dict)
        ranker.uniform_eval_setting(eval_dict=eval_dict)

        self.setup_eval(data_dict, eval_dict, model_para_dict)

        model_id = model_para_dict['model_id']
        fold_num, label_type, max_label = data_dict['fold_num'], data_dict['label_type'], data_dict['max_rele_level']
        train_presort, validation_presort, test_presort = \
            data_dict['train_presort'], data_dict['validation_presort'], data_dict['test_presort']
        # for quick access of common evaluation settings
        epochs, loss_guided = eval_dict['epochs'], eval_dict['loss_guided']
        vali_k, log_step, cutoffs = eval_dict['vali_k'], eval_dict['log_step'], eval_dict['cutoffs']
        do_vali, vali_metric, do_summary = eval_dict['do_validation'], eval_dict['vali_metric'], eval_dict['do_summary']
        cv_tape = CVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=do_vali)
        for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
            # TODO to be checked?
            ranker.init()  # initialize or reset with the same random initialization

            train_data, test_data, vali_data = self.load_data(eval_dict, data_dict, fold_k)

            if do_vali:
                vali_tape = ValidationTape(fold_k=fold_k, num_epochs=epochs, validation_metric=vali_metric,
                                           validation_at_k=vali_k, dir_run=self.dir_run)
            if do_summary:
                summary_tape = SummaryTape(do_validation=do_vali, cutoffs=cutoffs, label_type=label_type,
                                           train_presort=train_presort, test_presort=test_presort, gpu=self.gpu)
            if not do_vali and loss_guided:
                opt_loss_tape = OptLossTape(gpu=self.gpu)

            for epoch_k in range(1, epochs + 1):
                torch_fold_k_epoch_k_loss, stop_training = ranker.train(train_data=train_data, epoch_k=epoch_k,
                                                                        presort=train_presort, label_type=label_type)
                ranker.scheduler.step()  # adaptive learning rate with step_size=40, gamma=0.5

                if stop_training:
                    print('training is failed !')
                    break
                if (do_summary or do_vali) and (epoch_k % log_step == 0 or epoch_k == 1):  # stepwise check
                    if do_vali:  # per-step validation score
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

            if do_vali:  # using the fold-wise optimal model for later testing based on validation data
                ranker.load(vali_tape.get_optimal_path(), device=self.device)
                vali_tape.clear_fold_buffer(fold_k=fold_k)
            else:  # buffer the model after a fixed number of training-epoches if no validation is deployed
                fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/',
                            name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')

            cv_tape.fold_evaluation(model_id=model_id, ranker=ranker, test_data=test_data, max_label=max_label,
                                    fold_k=fold_k)
        ndcg_cv_avg_scores = cv_tape.get_cv_performance()
        return ndcg_cv_avg_scores

        # if log_explanatory_diagram:
        """
            feature_importance = ranker._compute_feature_importances(train_data)
            feature_number = np.arange(1, data_dict['num_features'] + 1)
            performance_list = [model_id + ' Fold-' + str(fold_k)]
            plt.figure(figsize=(27, 12))
            plt.xticks(feature_number)
            plt.xlabel("feature")
            plt.ylabel("weight")
            plt.title(performance_list)
            plt.plot(feature_importance)
            plt.savefig(self.dir_run + str(performance_list) + '.png')
            plt.close()
        """
        # if show_explanatory_diagram:
        # if log_explanatory_diagram:
        # plt.show()
        # else:
        # feature_importance = ranker._compute_feature_importances(train_data)
        # feature_number = np.arange(1, data_dict['num_features'] + 1)
        # performance_list = [model_id + ' Fold-' + str(fold_k)]
        # plt.figure(feature_importance, figsize=(18, 8))
        # plt.xticks(feature_number)
        # plt.xlabel("feature")
        # plt.ylabel("weight")
        # plt.title(performance_list)
        # plt.show()



    def point_run(self, debug=False, model_id=None, data_id=None, dir_data=None, dir_output=None,
                  dir_json=None, reproduce=False):
        """
        Perform one-time run based on given setting.
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """
        if dir_json is None:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_model_setting(debug=debug, model_id=model_id)
        else:
            data_eval_json = dir_json + 'Data_Eval.json'
            self.set_eval_setting(eval_json=data_eval_json)
            self.set_data_setting(data_json=data_eval_json)
            self.set_model_setting(model_id=model_id, dir_json=dir_json)

        data_dict = self.get_default_data_setting()
        eval_dict = self.get_default_eval_setting()
        model_para_dict = self.get_default_model_setting()

        self.declare_global(model_id=model_id)

        if reproduce:
            self.kfold_cv_reproduce(data_dict=data_dict, eval_dict=eval_dict, model_para_dict=model_para_dict)
        else:
            self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, model_para_dict=model_para_dict)

    def grid_run(self, model_id=None, dir_json=None, debug=False, data_id=None, dir_data=None,
                 dir_output=None):
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
            self.set_eval_setting(debug=debug, eval_json=data_eval_sf_json)
            self.set_model_setting(model_id=model_id, dir_json=dir_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_model_setting(debug=debug, model_id=model_id)

        self.declare_global(model_id=model_id)
        ''' select the best setting through grid search '''
        vali_k, cutoffs = 10, [1, 3, 5, 10, 20]
        max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
        k_index = cutoffs.index(vali_k)
        max_common_para_dict, max_model_para_dict = None, None

        for data_dict in self.iterate_data_setting():
            for eval_dict in self.iterate_eval_setting():
                assert self.eval_setting.check_consistence(vali_k=vali_k, cutoffs=cutoffs)  # a necessary consistence
                for model_para_dict in self.iterate_model_setting():
                    curr_cv_avg_scores = self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict,
                                                            model_para_dict=model_para_dict)
                    if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                        max_cv_avg_scores, max_eval_dict, max_model_para_dict = \
                            curr_cv_avg_scores, eval_dict, model_para_dict

        # log max setting
        self.log_max(data_dict=data_dict, eval_dict=max_eval_dict,
                     max_cv_avg_scores=max_cv_avg_scores,
                     log_para_str=self.model_parameter.to_para_string(log=True, given_para_dict=max_model_para_dict))

    def log_max(self, data_dict=None, max_cv_avg_scores=None, eval_dict=None, log_para_str=None):
        ''' Log the best performance across grid search and the corresponding setting '''
        dir_root, cutoffs = eval_dict['dir_root'], eval_dict['cutoffs']
        data_id = data_dict['data_id']

        data_eval_str = self.data_setting.to_data_setting_string(
            log=True) + '\n' + self.eval_setting.to_eval_setting_string(log=True)

        with open(file=dir_root + '/' + '_'.join([data_id, 'max.txt']),
                  mode='w') as max_writer:
            max_writer.write('\n\n'.join([data_eval_str, log_para_str,
                                          metric_results_to_string(max_cv_avg_scores, cutoffs, metric='nDCG')]))

    def run(self, debug=False, model_id=None, config_with_json=None, dir_json=None,
            data_id=None, dir_data=None, dir_output=None, grid_search=False, reproduce=False):
        if config_with_json:
            assert dir_json is not None
            if reproduce:
                self.point_run(debug=debug, model_id=model_id, dir_json=dir_json, reproduce=reproduce)
            else:
                self.grid_run(debug=debug, model_id=model_id, dir_json=dir_json)
        else:
            if grid_search:
                self.grid_run(debug=debug, model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output)
            else:
                self.point_run(debug=debug, model_id=model_id, data_id=data_id, dir_data=dir_data,
                               dir_output=dir_output, reproduce=reproduce)
