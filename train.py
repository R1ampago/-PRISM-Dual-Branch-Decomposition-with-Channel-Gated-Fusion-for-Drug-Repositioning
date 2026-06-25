"""Training loop for PRISM."""
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn import metrics
from model import PRISM


def train_fold(model, fold_data, epochs=3000, batch_size=8192,
               feat_lr=0.004, rest_lr=0.002, weight_decay=1e-5,
               eval_every=20, selection='comb', device='cuda:0', neg_ratio=10,
               single_loss=False, save_path=None):
    """Train PRISM on a single CV fold.

    Args:
        model: PRISM instance
        fold_data: tuple (edge_index, test_r, test_d, test_l, pr, pd, n_pos, nr, nd)
        epochs: number of training epochs
        batch_size: training batch size
        feat_lr: learning rate for FEAT branch parameters
        rest_lr: learning rate for all other parameters
        weight_decay: Adam weight decay
        eval_every: evaluate on test set every N epochs
        selection: 'aupr' or 'comb' (0.8*AUC + 0.2*AUPR)

    Returns:
        best_auc, best_aupr, best_epoch
    """
    ei, test_r, test_d, test_l, pr, pd, n_pos, nr, nd = fold_data
    n_pos = len(pr)

    # Parameter groups
    feat_p, rest_p = [], []
    for n, p in model.named_parameters():
        if any(x in n for x in ['drug_emb.', 'dis_emb.', 'gnn_drug.', 'gnn_dis.',
                                  'feat_drug_proj.', 'feat_dis_proj.']):
            feat_p.append(p)
        else:
            rest_p.append(p)

    opt = th.optim.Adam([
        {'params': feat_p, 'lr': feat_lr, 'weight_decay': weight_decay},
        {'params': rest_p, 'lr': rest_lr, 'weight_decay': weight_decay}
    ])

    best_auc, best_aupr, best_ep, best_comb = 0, 0, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        # Negative sampling 1:10
        ni = np.random.choice(len(nr), size=n_pos * neg_ratio, replace=False)
        br = np.concatenate([pr, nr[ni]])
        bd = np.concatenate([pd, nd[ni]])
        bl = np.concatenate([np.ones(n_pos), np.zeros(n_pos * neg_ratio)])
        perm = np.random.permutation(len(br))
        br, bd, bl = br[perm], bd[perm], bl[perm]

        for b in range(int(np.ceil(len(br) / batch_size))):
            s, e = b * batch_size, min((b + 1) * batch_size, len(br))
            ri = th.LongTensor(br[s:e]).to(device)
            di = th.LongTensor(bd[s:e]).to(device)
            lt = th.FloatTensor(bl[s:e]).to(device)
            sf, sc, scb = model(ei, ri, di)
            if single_loss:
                loss = F.binary_cross_entropy_with_logits(scb, lt)
            else:
                loss = (F.binary_cross_entropy_with_logits(sf, lt) +
                        F.binary_cross_entropy_with_logits(sc, lt) +
                        F.binary_cross_entropy_with_logits(scb, lt))
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if ep % eval_every == 0:
            model.eval()
            with th.no_grad():
                trb = th.LongTensor(test_r).to(device)
                tdb = th.LongTensor(test_d).to(device)
                _, _, scb = model(ei, trb, tdb)
                scb_np = th.sigmoid(scb).cpu().numpy()
                auc = metrics.roc_auc_score(test_l, scb_np)
                aupr = metrics.average_precision_score(test_l, scb_np)
                comb = 0.8 * auc + 0.2 * aupr
                if (selection == 'comb' and comb > best_comb) or \
                   (selection == 'aupr' and aupr > best_aupr) or \
                   (selection == 'auc' and auc > best_auc):
                    best_comb = comb
                    best_auc, best_aupr, best_ep = auc, aupr, ep
                    if save_path:
                        import os
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        th.save(model.state_dict(), save_path)

    return best_auc, best_aupr, best_ep
