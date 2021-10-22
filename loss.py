from typing import List, cast, NamedTuple, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from custom_types import IStateDict
from util import flatten_weight, split_state_dict
import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ITorchReturnTypeMax = NamedTuple('torch_return_type_max', [(
    'indices', torch.Tensor), ('values', torch.Tensor)])


def _icc_loss(pred: torch.Tensor, helper_preds: List[torch.Tensor]):
    kl_loss_helper = nn.KLDivLoss(reduction="batchmean")
    _sum = 0.0

    for helper_pred in helper_preds:
        _sum += kl_loss_helper(pred, helper_pred).float()

    return _sum / len(helper_preds)


def _transform_onehot(tensor: torch.Tensor) -> torch.Tensor:
    max_values = cast(torch.Tensor, torch.max(
        tensor, dim=1, keepdim=True).values)
    return (tensor >= max_values).float() - \
        torch.sigmoid(tensor - max_values).detach() + \
        torch.sigmoid(tensor - max_values)


def _calculate_pseudo_label(local_pred: torch.Tensor, helper_preds: List[torch.Tensor]):
    _sum = torch.zeros_like(local_pred)
    for pred in [local_pred, *helper_preds]:
        one_hot = _transform_onehot(pred)
        _sum += one_hot

    return torch.argmax(_sum, dim=1)


def _consistency_regularization(
    pred: torch.Tensor,
    pred_noised: torch.Tensor,
    helper_preds: List[torch.Tensor]
):

    pseudo_label = _calculate_pseudo_label(
        pred_noised, helper_preds).type(torch.LongTensor).to(device)

    pseudo_label_CE_loss = F.cross_entropy(
        pred_noised, pseudo_label)
    kl_loss = _icc_loss(pred, helper_preds)

    return pseudo_label_CE_loss + kl_loss


def iccs_loss(
    pred: torch.Tensor,
    pred_noised: torch.Tensor,
    helper_preds: List[torch.Tensor],
    lambda_iccs: float
):
    return _consistency_regularization(pred, pred_noised, helper_preds) * lambda_iccs


def regularization_loss(sigma: Iterable[Parameter], psi: Iterable[Parameter], lambda_l1: float, lambda_l2: float):
    sigma = list(sigma)
    psi = list(psi)

    loss = 0.0
    for idx in range(len(sigma)):
        loss += torch.sum(((sigma[idx] - psi[idx]) ** 2) * lambda_l2)
        loss += torch.sum(torch.abs(psi[idx]) * lambda_l1)

    return loss


def src_loss(local_last_feature_map: torch.Tensor, helper_last_feature_maps: List[torch.Tensor], mini_batch:int):
    # mean_feature_map vs local_last_feature_map
    
    # mean_feature_map = torch.mean(helper_last_feature_maps)
    shape = list(helper_last_feature_maps[0].size())
    total_feature_map = torch.empty(shape).to(device)
    for i in range(len(helper_last_feature_maps)):
       total_feature_map.add(helper_last_feature_maps[i])
       mean_feature_map = total_feature_map/len(helper_last_feature_maps)

    A_local = torch.reshape(local_last_feature_map,(mini_batch,-1))
    A_helper = torch.reshape(mean_feature_map,(mini_batch,-1))
    
    A_local_trans = torch.Tensor.transpose(A_local,0,1)
    A_helper_trans = torch.Tensor.transpose(A_helper,0,1)

    G1 = torch.mm(A_local, A_local_trans)
    G2 = torch.mm(A_helper, A_helper_trans)
    #shape1 = list(G1.size())
    #R1_inner = torch.empty(shape1).to(device)

    #shape2 = list(G2.size())
    #R2_inner = torch.empty(shape2).to(device)
    R1 = G1 * 1/G1.norm(dim = 1).reshape(-1,1)
    R2 = G2 * 1/G2.norm(dim = 1).reshape(-1,1)
    # list1 = []
    # for i in range(mini_batch):
    #     #G1[i] = torch.unsqueeze(G1[i],0)
    #     #G1[i] = torch.unsqueeze(G1[i],0)
    #     print(G1[i].size())
    #     G1_norm = F.normalize(G1[i], p=2.0, dim=1)  #, eps=1e-12, out=None
    #     G1_fraction = torch.div(G1[i],G1_norm)
    #     list1.append(G1_fraction)
    #     if i == mini_batch:
    #         R1_inner = list1

    # list2 = []
    # for i in range(mini_batch):
    #     #G2[i] = torch.unsqueeze(G2[i],0)
    #     #G2[i] = torch.unsqueeze(G2[i],0)
    #     G2_norm = F.normalize(G2[i], p=2.0, dim=1) #, p=2.0, dim=1, eps=1e-12, out=None
    #     G2_fraction = torch.div(G2[i],G2_norm)
    #     list2.append(G2_fraction)
    #     if i == mini_batch:
    #         R2_inner = list2   

    # R1 = torch.Tensor.transpose(R1_inner,0,1)
    # R2 = torch.Tensor.transpose(R2_inner,0,1)

    # mse of R1 & R2: sqrt[ (R1-R2) **2 ]
    return F.mse_loss(R1, R2) 