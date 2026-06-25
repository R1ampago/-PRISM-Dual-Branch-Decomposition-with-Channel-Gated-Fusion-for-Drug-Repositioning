"""PRISM: dual-branch drug-disease association with two-hop cross-attention."""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from layers import LowRankBottleneck, LightMultiScale


class PRISM(nn.Module):
    """PRISM: FEAT (multi-scale similarity aggregation) + CROSS (2-hop bipartite
       cross-attention) + channel-wise gate for dimension-level fusion."""

    def __init__(self, n_drugs, n_diseases, drug_adjs, dis_adjs,
                 hidden_dim=256, embed_dim=128, dropout=0.3, ch_dropout=0.5,
                 use_lowrank=True, bottleneck_rank=16, fusion_mode='channel_gate',
                 detach_gate=True, shared_embed=True):
        super().__init__()
        hd = hidden_dim
        ed = embed_dim
        self.nd = n_diseases
        self.nr = n_drugs
        self.d = hd
        self.use_lowrank = use_lowrank
        self.fusion_mode = fusion_mode
        self.detach_gate = detach_gate
        self.shared_embed = shared_embed

        # ── FEAT branch ──
        self.drug_emb = nn.Embedding(n_drugs, ed)
        self.dis_emb = nn.Embedding(n_diseases, ed)
        nn.init.xavier_uniform_(self.drug_emb.weight)
        nn.init.xavier_uniform_(self.dis_emb.weight)

        self.gnn_drug = LightMultiScale(ed, ed)
        self.gnn_dis = LightMultiScale(ed, ed)
        self.drug_adjs = drug_adjs
        self.dis_adjs = dis_adjs
        self.feat_drug_proj = nn.Linear(ed, hd)
        self.feat_dis_proj = nn.Linear(ed, hd)

        # FEAT decoder: 256→128→64→1
        self.fh = nn.Sequential(
            nn.Linear(hd, hd // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hd // 2, hd // 4), nn.ReLU()
        )
        self.hf_head = nn.Linear(hd // 4, 1)

        # ── CROSS branch ──
        if shared_embed:
            self.cross_drug_emb = None   # reuse self.drug_emb
            self.cross_dis_emb = None    # reuse self.dis_emb
        else:
            self.cross_drug_emb = nn.Embedding(n_drugs, ed)
            self.cross_dis_emb = nn.Embedding(n_diseases, ed)
            nn.init.xavier_uniform_(self.cross_drug_emb.weight)
            nn.init.xavier_uniform_(self.cross_dis_emb.weight)

        self.cross_drug_proj = nn.Linear(ed, hd)
        self.cross_dis_proj = nn.Linear(ed, hd)

        # Direction-specific value projection
        if use_lowrank:
            self.td2 = LowRankBottleneck(hd, bottleneck_rank)
            self.tr2 = LowRankBottleneck(hd, bottleneck_rank)
        else:
            self.td2 = nn.Linear(hd, hd, bias=False)
            self.tr2 = nn.Linear(hd, hd, bias=False)

        # Attention projections
        self.aqd = nn.Linear(hd, hd)
        self.aqp = nn.Linear(hd, hd)
        self.ak = nn.Linear(hd, hd)

        # Hop-2 gate residuals
        self.gate_d = nn.Sequential(nn.Linear(hd * 2, hd), nn.Sigmoid())
        self.gate_p = nn.Sequential(nn.Linear(hd * 2, hd), nn.Sigmoid())

        # CROSS decoder: 256→128→64→1 (stronger dropout)
        self.ch = nn.Sequential(
            nn.Linear(hd, hd // 2), nn.ReLU(), nn.Dropout(ch_dropout),
            nn.Linear(hd // 2, hd // 4), nn.ReLU()
        )
        self.hc_head = nn.Linear(hd // 4, 1)

        # ── Fusion ──
        self.fusion_mode = fusion_mode
        if fusion_mode == 'channel_gate':
            self.gate = nn.Sequential(
                nn.Linear(hd // 4 * 2, hd // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hd // 2, hd // 4)
            )
        elif fusion_mode == 'concat_linear':
            self.concat_fusion = nn.Linear(hd // 4 * 2, hd // 4)
        elif fusion_mode == 'scalar_gate':
            self.scalar_gate = nn.Sequential(
                nn.Linear(hd // 4 * 2, hd // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hd // 2, 1)
            )
        self.hcm = nn.Linear(hd // 4, 1)

    def forward(self, edge_index, drug_idx, dis_idx):
        """Forward pass.
        Args:
            edge_index: (drug_ids, dis_ids) of training edges
            drug_idx:   drug indices for current batch
            dis_idx:    disease indices for current batch
        Returns:
            sf:  FEAT branch logits
            sc:  CROSS branch logits
            scb: Combined logits
        """
        ds, dd = edge_index

        # Move KNN adjs to device once (instead of per-call in LightMultiScale)
        device = ds.device
        if self.drug_adjs[0].device != device:
            self.drug_adjs = [a.to(device) for a in self.drug_adjs]
            self.dis_adjs = [a.to(device) for a in self.dis_adjs]

        # ── FEAT ──
        fd = self.feat_drug_proj(self.gnn_drug(self.drug_emb.weight, self.drug_adjs))
        fp = self.feat_dis_proj(self.gnn_dis(self.dis_emb.weight, self.dis_adjs))
        ife = fd[drug_idx] * fp[dis_idx]
        hf = self.fh(ife)

        # ── CROSS ──
        cd = self.cross_drug_proj(self.drug_emb.weight if self.shared_embed else self.cross_drug_emb.weight)
        cp = self.cross_dis_proj(self.dis_emb.weight if self.shared_embed else self.cross_dis_emb.weight)

        # Hop 1
        co1 = self._bipartite_attn(self.aqp(cp), self.tr2(cd[ds]), dd, self.nd)
        do1 = self._bipartite_attn(self.aqd(cd), self.td2(cp[dd]), ds, self.nr)

        # Hop 2
        do2 = self._bipartite_attn(self.aqd(cd), self.td2(co1[dd]), ds, self.nr)
        co2 = self._bipartite_attn(self.aqp(cp), self.tr2(do1[ds]), dd, self.nd)

        # Gate residual
        do2 = do2 + self.gate_d(th.cat([do2, do1], -1)) * do1
        co2 = co2 + self.gate_p(th.cat([co2, co1], -1)) * co1

        do = do1 + do2
        co = co1 + co2

        icr = do[drug_idx] * co[dis_idx]
        hc = self.ch(icr)

        # ── Fusion ──
        if self.fusion_mode == 'channel_gate':
            gate_input = th.cat([hf.detach(), hc.detach()], -1) if self.detach_gate else th.cat([hf, hc], -1)
            g = th.sigmoid(self.gate(gate_input))
            fu = g * hf + (1 - g) * hc
        elif self.fusion_mode == 'concat_linear':
            fu = self.concat_fusion(th.cat([hf, hc], -1))
        elif self.fusion_mode == 'scalar_gate':
            gate_input = th.cat([hf.detach(), hc.detach()], -1) if self.detach_gate else th.cat([hf, hc], -1)
            g = th.sigmoid(self.scalar_gate(gate_input))  # [B, 1]
            fu = g * hf + (1 - g) * hc

        sf = self.hf_head(hf).squeeze(-1)
        sc = self.hc_head(hc).squeeze(-1)
        scb = self.hcm(fu).squeeze(-1)

        return sf, sc, scb

    def _bipartite_attn(self, query_proj, message, dst_idx, dst_size):
        """Direction-separated dot-product attention with per-destination softmax."""
        key = self.ak(message)
        qp = query_proj[dst_idx]
        raw = (qp * key).sum(-1) / (self.d ** 0.5)

        mx = th.zeros(dst_size, device=message.device).index_reduce_(
            0, dst_idx, raw, 'amax', include_self=False)
        ee = (raw - mx[dst_idx]).exp()
        es = th.zeros(dst_size, device=message.device).index_add_(0, dst_idx, ee)
        alpha = ee / es[dst_idx].clamp(1e-8)

        out = th.zeros(dst_size, message.size(1), device=message.device)
        return out.index_add_(0, dst_idx, alpha.unsqueeze(1) * message)

    def predict(self, edge_index, drug_idx, dis_idx):
        """Return combined logits only (for inference)."""
        _, _, scb = self.forward(edge_index, drug_idx, dis_idx)
        return scb
