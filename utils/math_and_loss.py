import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def rmse(y_hat, y, norm_constant=1):
    return 100 * np.sqrt(np.mean(np.square(y_hat - y))) / norm_constant

def sigmoid_weight(z):
    return 1 / (1 + torch.exp(15 * (z - 0.5)))

def exp_weight(z):
    return torch.exp(2 * (-z))

def inv_weight(z, d=-0.1):
    eps = 1e-6
    return 0.3*(z+eps)**(d)

CD_WEIGHT_FN = {
    'sig': sigmoid_weight,
    'exp': exp_weight,
    'inv': inv_weight,
}

DIAG_REG = {
    'l1': torch.abs, 
    'l2': torch.square
}


class JSDivLoss(nn.Module):

    def __init__(self, reduction='none'):
        super(JSDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inp, trg):
        '''
        inp : pred
        trg : label
        '''
        bsz = inp.shape[0]

        # avoid total loss NaN
        eps = 1e-7
        inp = inp + eps
        trg = trg + eps
        
        dist_mean = (0.5 * (inp + trg)).log()

        inp_given_mean = F.kl_div(dist_mean, inp.log(), reduction='none', log_target=True)
        trg_given_mean = F.kl_div(dist_mean, trg.log(), reduction='none', log_target=True)
        loss = 0.5 * (inp_given_mean + trg_given_mean)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'batchmean':
            loss = loss.sum() / bsz
        elif self.reduction == 'sum':
            loss = loss.sum()
        else: 
            loss = loss
        
        return loss


class ChiSquareLoss(nn.Module):
    
    def __init__(self, reduction='none'):
        super(ChiSquareLoss, self).__init__()
        
        self.reduction = reduction

    def forward(self, inp, trg):
        '''
        inp : pred
        trg : label
        '''
        bsz = inp.shape[0]

        eps=1e-7
        neg_log = -(-trg/2-1.442695*torch.lgamma(trg/2 + eps)-inp/2*1.442695 + (trg/2-1)*torch.log2(inp+eps))

        if self.reduction == 'mean':
            loss = neg_log.mean()
        elif self.reduction == 'batchmean':
            loss = neg_log.sum() / bsz
        elif self.reduction == 'sum':
            loss = neg_log.sum()
        else:
            loss = loss

        return loss


class LossMeter:

    def __init__(self, task, mat_lambda, K=1, edloss='mse', nwloss='jsd', ed_cdw=False, nw_cdw=False, 
                 cdw_mthd='sig', norm_const=1, device=None):
        super(LossMeter, self).__init__()

        self.task = task
        self.K = K
        self.mat_lambda = mat_lambda

        self._edloss = edloss
        self._nwloss = nwloss
        self._norm_const = norm_const # seq len norm const for chi2 regression

        self._ed_cdw = True if ed_cdw == 'True' else False
        self._nw_cdw = True if nw_cdw == 'True' else False

        self.cdw_method = cdw_mthd
        self.cdw_fn = CD_WEIGHT_FN[self.cdw_method]

        if self.task == 'vanilla':
            if edloss == 'chi2':
                self.ed_loss = ChiSquareLoss(reduction='sum').to(device)
            else:
                self.ed_loss = nn.MSELoss(reduction='sum').to(device)
            self.nw_loss = None

        elif self.task == 'ednw':
            if edloss == 'chi2':
                self.ed_loss = ChiSquareLoss(reduction='sum').to(device)
            else:
                self.ed_loss = nn.MSELoss(reduction='sum').to(device)

            if nwloss == 'mse':
                self.nw_loss = nn.MSELoss(reduction='sum').to(device)
            elif nwloss == 'jsd':
                self.nw_loss = JSDivLoss(reduction='sum').to(device)
        else:
            self.ed_loss = None
            if nwloss == 'mse':
                self.nw_loss = nn.MSELoss(reduction='sum').to(device)
            elif nwloss == 'jsd':
                self.nw_loss = JSDivLoss(reduction='sum').to(device)
        
        self.device = device
        self._train = True

        self.total_sum = 0
        self.ed_sum = 0
        self.nw_sum = 0
        self.count = 0

    def backprop(self, ed_predict, ed_label, attn_predict=None, attn_label=None, count=1):

        batch_total_loss = self.update(ed_predict, ed_label, attn_predict, attn_label, count)
        batch_total_loss.backward()

    def update(self, ed_predict, ed_label, attn_predict=None, attn_label=None, count=1):
        
        bsz = ed_predict.shape[0]

        if self._edloss == 'chi2':
            ed_predict = ed_predict * self._norm_const
            ed_label = ed_label * self._norm_const

        if self._ed_cdw or self._nw_cdw:
            cd_weight = self.cdw_fn(ed_label)

            cdw_dim = [1 for _ in range(attn_predict.dim())]
            cdw_dim[0] = bsz

        # 'vanilla(ed) loss
        if self.task == 'vanilla':
            if self._ed_cdw and self._train:
                batch_ed_loss = self.ed_loss(ed_predict * cd_weight, ed_label * cd_weight)
            else:
                batch_ed_loss = self.ed_loss(ed_predict, ed_label)

            self.ed_sum += batch_ed_loss.sum().item()    
            batch_total_loss = batch_ed_loss
        
        # 'ednw' loss
        elif self.task == 'ednw':
            # edit distance
            if self._ed_cdw and self._train:
                batch_ed_loss = self.ed_loss(ed_predict * cd_weight, ed_label * cd_weight)
            else:
                batch_ed_loss = self.ed_loss(ed_predict, ed_label)

            self.ed_sum += batch_ed_loss.sum().item()
            
            # attention
            if self._nw_cdw and self._train:
                cd_weight = cd_weight.reshape(cdw_dim)
                batch_nw_loss = self.nw_loss(attn_predict * cd_weight, attn_label * cd_weight)
            else:
                batch_nw_loss = self.nw_loss(attn_predict, attn_label)

            self.nw_sum += batch_nw_loss.sum().item()
            batch_total_loss = batch_ed_loss + self.mat_lambda * batch_nw_loss

        self.total_sum += batch_total_loss.item()
        self.count += count

        return batch_total_loss

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def zero_sum(self):

        self.total_sum = 0
        self.ed_sum = 0
        self.nw_sum = 0
        self.count = 0
  
    def get_avg(self):
        total_avg = self.total_sum/self.count
        ed_avg = self.ed_sum/self.count
        nw_avg = self.nw_sum/self.count
        
        return total_avg, ed_avg, nw_avg

