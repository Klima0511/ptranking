
import torch

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.adhoc.adhoc_metric import torch_ndcg_at_k
from ptranking.metric.metric_utils import get_delta_ndcg

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
        batch_predict_rankings = batch_predict_rankings.cpu()
        batch_ideal_rankings =  batch_ideal_rankings.cpu()


        ndcg_at_k = torch_ndcg_at_k(batch_predict_rankings=batch_predict_rankings.view(1, -1),
                                    batch_ideal_rankings=batch_ideal_rankings.view(1, -1), k=1,
                                    label_type=LABEL_TYPE.MultiLabel)
        sum_ndcg_at_k = torch.add(sum_ndcg_at_k, torch.squeeze(ndcg_at_k))
        cnt += 1

    avg_ndcg_at_k = sum_ndcg_at_k / cnt
    #print('avg_ndcg_at_k', avg_ndcg_at_k.size())
    #return avg_ndcg_at_k.numpy()
    return avg_ndcg_at_k.item()

def lambdarank_objective(yhat, y, sample_weight=None, **kwargs):
    gradient = torch.ones_like(yhat)
    hessian = torch.ones_like(yhat)

    head = 0
    sigma = 1.0
    group = kwargs['group']
    print(group.shape)
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
        batch_stds = batch_stds.cpu()
        batch_stds_sorted_via_preds = batch_stds_sorted_via_preds.cpu()

        # pairwise weighting based on delta-nDCG
        batch_delta_ndcg = get_delta_ndcg(batch_ideal_rankings=batch_stds,
                                          batch_predict_rankings=batch_stds_sorted_via_preds, label_type=LABEL_TYPE.MultiLabel,
                                          device='cpu')
        batch_delta_ndcg = batch_delta_ndcg.cuda(1)
        print("1")
        _batch_grad_order1 *= batch_delta_ndcg
        _batch_grad_order1_triu = torch.triu(_batch_grad_order1, diagonal=1)
        batch_grad_order1 = _batch_grad_order1_triu + torch.transpose(-1.0 * _batch_grad_order1_triu, dim0=1, dim1=2)
        print("2")
        _batch_grad_order2 *= batch_delta_ndcg
        _batch_grad_order2_triu = torch.triu(_batch_grad_order2, diagonal=1)
        batch_grad_order2 = _batch_grad_order2_triu + torch.transpose(-1.0 * _batch_grad_order2_triu, dim0=1, dim1=2)
        print("3")
        # print('grad_order1', grad_order1.size())
        batch_grad_order1 = torch.sum(batch_grad_order1, 1)
        # print('grad_order1', grad_order1.size())
        batch_grad_order2 = torch.sum(batch_grad_order2, 1)
        print("4")
        gradient[head:head + gr] = torch.squeeze(batch_grad_order1)
        hessian[head:head + gr]  = torch.squeeze(batch_grad_order2)

        head += gr
    print("5")
    return gradient, hessian