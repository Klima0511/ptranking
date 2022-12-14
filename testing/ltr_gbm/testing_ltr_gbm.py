
import torch

import numpy as np

from ptranking.ltr_global import ltr_seed
from ptranking.ltr_gbm.eval.ltr_gbm import GBMLTREvaluator

np.random.seed(seed=ltr_seed)
torch.manual_seed(seed=ltr_seed)

if __name__ == '__main__':

    """
    >>> Learning-to-Rank Models based on gradient boosting machine <<< 

    -----------------------------------------------------------------------------------------
    | GrowNet | GBDTRanker |                                                         

    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_LTR | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_LTR | Istella_S | Istella | Istella_X                                         |
    -----------------------------------------------------------------------------------------
    | IRGAN_MQ2008_Semi                                                                      |
    -----------------------------------------------------------------------------------------
    """

    cuda = None  # the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu

    debug = True  # in a debug mode, we just check whether the model can operate

    sf_id = 'gbdt'  # pointsf | listsf | gbdt, namely the type of scoring function

    config_with_json = True  # specify configuration with json files or not

    models_to_run = [
        'GBDTRanker'
        #'PGBMRanker'
    ]

    evaluator = GBMLTREvaluator(cuda=cuda)

    if config_with_json:  # specify configuration with json files
        # the directory of json files
        # dir_json = '/Users/dryuhaitao/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adversarial/json/'
        # dir_json = '/home/dl-box/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adversarial/json/'
        # dir_json = '/Users/iimac/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/json/iimac/'
        dir_json = '/Users/iimac/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/StudentFork/Tabnet_YangKaiyu/TabPTRanking/testing/ltr_gbm/json/'

        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, config_with_json=config_with_json, dir_json=dir_json)

    else:  # specify configuration manually
        data_id = 'MQ2008_Super'

        ''' Location of the adopted data '''
        # dir_data = '/Users/dryuhaitao/WorkBench/Corpus/' + 'LETOR4.0/MQ2008/'
        # dir_data = '/home/dl-box/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'
        #dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'
        dir_data = '/Users/iimac/Workbench/Corpus/L2R/LETOR4.0/MQ2008/'

        ''' Output directory '''
        # dir_output = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/ALTR/'
        # dir_output = '/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/Listwise/'
        #dir_output = '/Users/solar/WorkBench/CodeBench/PyCharmProject/Project_output/Out_L2R/'
        dir_output = '/Users/iimac/Workbench/CodeBench/Output/NeuralLTR/'

        grid_search = False  # with grid_search, we can explore the effects of different hyper-parameters of a model

        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output,
                          sf_id=sf_id, grid_search=grid_search, reproduce=False)
