
import torch

import matplotlib.pyplot as plt

from ptranking.ltr_gbm.base.gbdt.gbdt import GBDT
from ptranking.data.data_utils import SPLIT_TYPE, LABEL_TYPE, GBMDataset
from ptranking.metric.adhoc.adhoc_metric import torch_ndcg_at_k
from ptranking.metric.metric_utils import get_delta_ndcg

#%% Objective for gbdt: pointwise-mse
def mseloss_objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    #print('yhat', yhat)
    loss = (yhat - y).pow(2).mean().sqrt()
    return loss

#%% Objective for gbdt: listwise-lambdarank
def lambdarank_objective(yhat, y, sample_weight=None, **kwargs):
    gradient = torch.ones_like(yhat)
    hessian = torch.ones_like(yhat)

    head = 0
    sigma = 1.0
    group = kwargs['group']
    for i in range(group.shape[0]):
        #print('gr ==')
        gr = int(group[i])
        batch_stds = y[head:head + gr].view(1, -1)
        ngbm_preds = yhat[head:head + gr].view(1, -1)

        batch_preds_sorted, batch_preds_sorted_inds = torch.sort(ngbm_preds, dim=1, descending=True)  # sort documents according to the predicted relevance
        batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)  # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

        batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) -\
                          torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
        batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}

        # batch_std_p_ij = 0.5 * (1.0 + batch_std_Sij)

        batch_s_ij = torch.unsqueeze(batch_preds_sorted, dim=2) -\
                     torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j
        # batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)
        # batch_p_ij = torch.sigmoid(self.sigma * batch_s_ij)

        component = 1.0 / (1.0 + torch.exp(sigma * batch_s_ij))
        _batch_grad_order1 = sigma * (0.5 * (1 - batch_std_Sij) - component)
        _batch_grad_order2 = sigma * sigma * component * (1.0 - component)

        # pairwise weighting based on delta-nDCG
        batch_delta_ndcg = get_delta_ndcg(batch_ideal_rankings=batch_stds,
                                          batch_predict_rankings=batch_stds_sorted_via_preds, label_type=LABEL_TYPE.MultiLabel,
                                          device='cpu')
        _batch_grad_order1 *= batch_delta_ndcg
        _batch_grad_order1_triu = torch.triu(_batch_grad_order1, diagonal=1)
        batch_grad_order1 = _batch_grad_order1_triu + torch.transpose(-1.0 * _batch_grad_order1_triu, dim0=1, dim1=2)

        _batch_grad_order2 *= batch_delta_ndcg
        _batch_grad_order2_triu = torch.triu(_batch_grad_order2, diagonal=1)
        batch_grad_order2 = _batch_grad_order2_triu + torch.transpose(-1.0 * _batch_grad_order2_triu, dim0=1, dim1=2)

        # print('grad_order1', grad_order1.size())
        batch_grad_order1 = torch.sum(batch_grad_order1, 1)
        # print('grad_order1', grad_order1.size())
        batch_grad_order2 = torch.sum(batch_grad_order2, 1)

        gradient[head:head + gr] = torch.squeeze(batch_grad_order1)
        hessian[head:head + gr]  = torch.squeeze(batch_grad_order2)

        head += gr

    return gradient, hessian




def nDCG_metric(yhat, y, sample_weight=None, **kwargs):
    #print('yhat', yhat)
    head = 0
    cnt = 0
    sum_ndcg_at_k = torch.zeros(1)
    #group = group.astype(np.int).tolist()
    #print('group', group.shape[0])
    #print('y', y.size())
    #for gr in group:
    group = kwargs['group']
    for i in range(group.shape[0]):
        #print('gr ==')
        gr = int(group[i])
        tor_per_query_std_labels = y[head:head + gr]
        tor_per_query_preds = yhat[head:head + gr]
        head += gr

        _, tor_sorted_inds = torch.sort(tor_per_query_preds, descending=True)

        batch_predict_rankings = tor_per_query_std_labels[tor_sorted_inds]
        batch_ideal_rankings, _ = torch.sort(tor_per_query_std_labels, descending=True)

        ndcg_at_k = torch_ndcg_at_k(batch_predict_rankings=batch_predict_rankings.view(1, -1),
                                    batch_ideal_rankings=batch_ideal_rankings.view(1, -1), k=1,
                                    label_type=LABEL_TYPE.MultiLabel)
        sum_ndcg_at_k = torch.add(sum_ndcg_at_k, torch.squeeze(ndcg_at_k))
        cnt += 1

    avg_ndcg_at_k = sum_ndcg_at_k / cnt
    #print('avg_ndcg_at_k', avg_ndcg_at_k.size())
    #return avg_ndcg_at_k.numpy()
    return avg_ndcg_at_k.item()


if __name__ == '__main__':
    #%% Load data
    data_id = 'MQ2008_Super'
    dir_data = '/Users/iimac/Workbench/Corpus/L2R/LETOR4.0/MQ2008/'
    fold_k = 1
    fold_k_dir = dir_data + 'Fold' + str(fold_k) + '/'
    file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'
    train_data = GBMDataset(split_type=SPLIT_TYPE.Train, file=file_train, data_id=data_id, buffer=False).get_data()
    vali_data =  GBMDataset(split_type=SPLIT_TYPE.Validation, file=file_vali, data_id=data_id, buffer=False).get_data()
    X_test, y_test, group_test =  GBMDataset(split_type=SPLIT_TYPE.Test, file=file_test, data_id=data_id, buffer=False).get_data()

    # Train on set
    model = GBDT()

    #model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)

    #model.train(train_data, objective=mseloss_objective, metric=nDCG_metric, group=train_data[2])

    #model.train(train_data, objective=lambdarank_objective, metric=nDCG_metric, ranking_obj=True, ranking_metric=True)

    model.train(train_data, objective=lambdarank_objective, metric=nDCG_metric, valid_set=vali_data, ranking_obj=True, ranking_metric=True)


    #% Point and probabilistic predictions. By default, 100 probabilistic estimates are created
    yhat_point = model.predict(X_test)
    #yhat_dist = model.predict_dist(X_test)

    # Scoring
    #rmse = model.metric(yhat_point, y_test)
    rmse = model.metric(yhat_point, model._convert_array(y_test), None, group=group_test)
    #crps = model.crps_ensemble(yhat_dist, y_test).mean()
    # Print final scores
    print(f'RMSE GBDT: {rmse:.2f}')
    #print(f'CRPS PGBM: {crps:.2f}')

    #%% Plot all samples
    plt.rcParams.update({'font.size': 22})
    plt.plot(y_test, 'o', label='Actual')
    plt.plot(yhat_point.cpu(), 'ko', label='Point prediction PGBM')
    #plt.plot(yhat_dist.cpu().max(dim=0).values, 'k--', label='Max bound PGBM')
    #plt.plot(yhat_dist.cpu().min(dim=0).values, 'k--', label='Min bound PGBM')
    plt.legend()
    #plt.show()
