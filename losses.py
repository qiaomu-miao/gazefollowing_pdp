import torch
import torch.nn as nn

class KL_div_modified(nn.Module):
    def __init__(self, epsilon=2.2204e-16, reduction='batchmean'):
        # eps value is adopted from the paper "What do different evaluation metrics tell us about saliency models" which says in the MIT saliency benchmark that eps is 2.2204e-16
        super(KL_div_modified, self).__init__()
        self.eps = epsilon
        self.reduction=reduction
    def forward(self, input, target):
        kl_div = target * (torch.log(self.eps+torch.divide(target, input+self.eps)))
        if self.reduction=='batchmean':
            kl_div = torch.mean(kl_div.sum(dim=1))
        elif self.reduction=='sum':
            kl_div = torch.sum(kl_div)
        elif self.reduction=='none':
            kl_div = kl_div
        return kl_div
