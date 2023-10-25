#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description
"""

import os

import numpy as np

import ptranking.ltr_node.eval.ltr_node
from ptranking.ltr_global import ltr_seed

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
np.random.seed(seed=ltr_seed)

if __name__ == '__main__':

    """
    >>> Tree-based Learning-to-Rank Models <<<

    (3) Tree-based Model
    -----------------------------------------------------------------------------------------
    | LightGBMLambdaMART                                                                    |
    -----------------------------------------------------------------------------------------

    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   |  %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_LTR | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_LTR | Istella_S | Istella | Istella_X                                         |
    -----------------------------------------------------------------------------------------

    """

    cuda = 1  # the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu
    # cuda = 1
    debug = False  # in a debug mode, we just check whether the model can operate

    config_with_json = True  # specify configuration with json files or not

    reproduce = False

    models_to_run = [
        # 'SoftRank',
        # 'RankMSE',
        # 'TabNet',
        # 'LambdaRank',
        # 'ListNet',
        # 'ListMLE',
        # 'RankCosine',
        # 'ApproxNDCG',
        # 'WassRank',
        # 'STListNet',
        # 'LambdaLoss',
        # 'MDPRank',
        # 'ExpectedUtility',
        # 'DASALC',
        # 'HistogramAP',
        'node',
        # 'TwinRank'
    ]

    evaluator = ptranking.ltr_node.eval.ltr_node.NeuralDecisionEnsemblesLTREvaluator(cuda=cuda)


    if config_with_json:  # specify configuration with json files
        # the directory of json files
        # dir_json = '/Users/dryuhaitao/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'
        # dir_json = '/Users/solar/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/'
        # dir_json = '/Users/iimac/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/drl_ptranking/testing/ltr_adhoc/json/'

        # dir_json = '/Users/solar/WorkBench/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/json/solar/'
        # dir_json = '/Users/iimac/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/json/iimac/'
        # dir_json = '/Users/iilab/PycharmProjects/ptranking/ptranking/ltr_ntree/eval/json/'
        dir_json = '/home/user/Workbench/tan_haonan/test/testing/ltr_node/json/'

        # test_bt_bn_opt
        # dir_json = '/home/user/T2_Workbench/ExperimentBench/test_bt_bn_opt/'
        # dir_json = '/home/user/T2_Workbench/ExperimentBench/test_bt_bn_opt_mq2008/'

        # dir_json = '/home/user/T2_Workbench/ExperimentBench/test_bt_bn_opt/Tmp_results/'

        # MetricEM baseline
        # dir_json = '/home/user/T2_Workbench/ExperimentBench/MetricEM_ms30k_Baseline_300_GE/'

        # TwinRank on Istella_X
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank/R/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank/GE/'

        # TwinRank baseline on set1
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank/baseline_set1/R/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank/baseline_set1/GE/'

        # TwinRank baseline on ms30k
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank/baseline_ms30k/R/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank/baseline_ms30k/GE/'

        # TwinRank reproduce on ms30k
        # TwinSigST
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/TwinSigST/P/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/TwinSigST/AP/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/TwinSigST/nDCG/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/TwinSigST/nERR/'

        # SignTwinSig
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/SignTwinSig/P/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/SignTwinSig/AP/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/SignTwinSig/nDCG/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/SignTwinSig/nERR/'

        # SignTwinSigAmp
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/SignTwinSigAmp/P/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/SignTwinSigAmp/AP/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/SignTwinSigAmp/nDCG/'
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/SignTwinSigAmp/nERR/'

        # listnet
        # dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank/reproduce_listnet/'

        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, config_with_json=config_with_json, dir_json=dir_json,
                          reproduce=reproduce)

    else:  # specify configuration manually
        ''' pointsf | listsf, namely the type of neural scoring function '''
        # sf_id = 'pointsf'

        ''' Selected dataset '''
        # data_id = 'Set1'
        data_id = 'MSLRWEB30K'
        #data_id = 'MQ2008_Super'

        ''' By grid_search, we can explore the effects of different hyper-parameters of a model '''
        grid_search = False

        ''' Location of the adopted data '''
        # dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2008/'
        dir_data = '/data/Corpus/MSLR-WEB30K/'
        # dir_data = '/home/user/T2_Workbench/Corpus/L2R/MSLR-WEB30K/'
        # dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'
        #dir_data = '/Users/iilab/Workbench/Data/MSLR-WEB30K/'

        #dir_data = "C:\\Users\\59799\\Desktop\\MQ2008\\"
        # dir_data = '/Users/iilab/Workbench/Data/MQ2008/'
        ''' Output directory '''
        # dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/'
        # dir_output = '/home/user/T2_Workbench/Project_output/Out_L2R/Listwise/'
        dir_output = '/data/tan_haonan/Output/node/'
        #dir_output = "C:\\Users\\59799\\Desktop\\MQ2008\\output\\"
        # dir_output = '/home/user/T2_Workbench/ExperimentBench/test_bt_bn_opt/Tmp_results/'
        # dir_output = '/home/user/T2_Workbench/ExperimentBench/LTR_Adversarial/Results/'

        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, grid_search=grid_search,
                          data_id=data_id, dir_data=dir_data, dir_output=dir_output, reproduce=reproduce)