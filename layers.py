"""PRISM model components: LowRankBottleneck, multi-scale LightGCN."""
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class LowRankBottleneck(nn.Module):
    """Low-rank bottleneck: V(U(x)), rank=16. U frozen optional, V trainable."""
    def __init__(self, d, rank=16):
        super().__init__()
        self.U = nn.Linear(d, rank, bias=False)
        self.V = nn.Linear(rank, d, bias=False)

    def forward(self, x):
        return self.V(self.U(x))



class LightMultiScale(nn.Module):
    """Multi-scale LightGCN with learned scale attention (K=2, 4, 8)."""
    def __init__(self, in_dim, out_dim, k_list=(2, 4, 8)):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.scale_att = nn.Linear(in_dim, len(k_list))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.constant_(self.W.bias, 0)
        nn.init.xavier_uniform_(self.scale_att.weight)
        nn.init.constant_(self.scale_att.bias, 0)

    def forward(self, x, adjs):
        # adjs are already on the correct device (moved once in PRISM.forward)
        outs = [F.relu(self.W(a @ x)) for a in adjs]
        stacked = th.stack(outs, dim=1)
        att = F.softmax(self.scale_att(x), dim=-1)
        return (att.unsqueeze(-1) * stacked).sum(dim=1)
