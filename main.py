"""PRISM: 10-fold cross-validation runner."""
import os
import sys
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch as th

from loader_prism import PRISMLoader
from model import PRISM
from train import train_fold
from utils import knn_csr


def main():
    parser = argparse.ArgumentParser(description='PRISM: Drug Repositioning')
    parser.add_argument('--dataset', type=str, default='Gdataset',
                        choices=['LRSSL', 'Ldataset', 'Gdataset', 'Cdataset'])
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=77)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--feat-lr', type=float, default=0.004)
    parser.add_argument('--rest-lr', type=float, default=0.002)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--ch-dropout', type=float, default=0.4)
    parser.add_argument('--neg-ratio', type=int, default=10)
    parser.add_argument('--selection', type=str, default='comb',
                        choices=['aupr', 'auc', 'comb'])
    parser.add_argument('--num-seeds', type=int, default=1)
    parser.add_argument('--save-dir', type=str, default='results')
    args = parser.parse_args()

    device = th.device(args.device if th.cuda.is_available() else 'cpu')
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ──
    loader = PRISMLoader(args.dataset)
    n_drugs, n_dis = loader.n_drug, loader.n_dis
    sd, sp_sim = loader.drug_sim, loader.dis_sim

    # ── KNN adjacencies ──
    cache_adj = f'cache/cache_{args.dataset}_adj.pkl'
    if os.path.exists(cache_adj):
        with open(cache_adj, 'rb') as f:
            pre = pickle.load(f)
        drug_adjs, dis_adjs = pre['adjs_d'], pre['adjs_p']
    else:
        drug_adjs, dis_adjs = [], []
        for k in [2, 4, 8]:
            adj_d = knn_csr(sd, k)
            drug_adjs.append(th.sparse_csr_tensor(
                adj_d.indptr.astype(np.int64),
                adj_d.indices.astype(np.int64),
                adj_d.data.astype(np.float32),
                size=(n_drugs, n_drugs)))
            adj_p = knn_csr(sp_sim, k)
            dis_adjs.append(th.sparse_csr_tensor(
                adj_p.indptr.astype(np.int64),
                adj_p.indices.astype(np.int64),
                adj_p.data.astype(np.float32),
                size=(n_dis, n_dis)))
        with open(cache_adj, 'wb') as f:
            pickle.dump({'adjs_d': drug_adjs, 'adjs_p': dis_adjs}, f)

    print(f"PRISM | {args.dataset} | {args.folds}-fold x {args.num_seeds} seeds | ch_dr={args.ch_dropout} {args.selection}-sel {args.epochs}ep")

    all_auc, all_aupr = [], []
    for si in range(args.num_seeds):
        seed = args.seed + si * 100
        th.manual_seed(seed); np.random.seed(seed)
        fold_auc, fold_aupr = [], []
        for cv in range(args.folds):
            edges, (test_r, test_d, test_l), (pr, pd, n_pos, nr, nd), _ = loader.get_fold(cv)
            ei = th.stack([th.LongTensor(edges[:, 0]).to(device),
                           th.LongTensor(edges[:, 1]).to(device)])
            model = PRISM(n_drugs, n_dis, drug_adjs, dis_adjs,
                          hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
                          dropout=args.dropout, ch_dropout=args.ch_dropout).to(device)
            fold_data = (ei, test_r, test_d, test_l, pr, pd, n_pos, nr, nd)
            save_path = f'cache/ckpt_{args.dataset}_seed{seed}/fold{cv}.pth'
            auc, aupr, ep = train_fold(model, fold_data, epochs=args.epochs,
                batch_size=args.batch_size, feat_lr=args.feat_lr, rest_lr=args.rest_lr,
                weight_decay=args.wd, selection=args.selection,
                neg_ratio=args.neg_ratio, device=args.device,
                save_path=save_path)
            fold_auc.append(auc); fold_aupr.append(aupr)
            print(f"  S{si+1}F{cv+1:2d}: AUC={auc:.4f} AUPR={aupr:.4f} @E{ep}", flush=True)
        all_auc.extend(fold_auc); all_aupr.extend(fold_aupr)
        print(f"  Seed{seed}: AUC={np.mean(fold_auc):.4f}+-{np.std(fold_auc):.4f}  AUPR={np.mean(fold_aupr):.4f}+-{np.std(fold_aupr):.4f}", flush=True)

    overall_auc = np.mean(all_auc); overall_std = np.std(all_auc)
    overall_p = np.mean(all_aupr); overall_pstd = np.std(all_aupr)
    print(f"\nOVERALL: AUC={overall_auc:.4f}+-{overall_std:.4f}  AUPR={overall_p:.4f}+-{overall_pstd:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(f'{args.save_dir}/{args.dataset}_results.txt', 'w') as f:
        f.write(f"PRISM | {args.dataset} | {args.folds}f x {args.num_seeds} seeds\n")
        f.write(f"AUROC: {overall_auc:.4f} +- {overall_std:.4f}\n")
        f.write(f"AUPR:  {overall_p:.4f} +- {overall_pstd:.4f}\n")


if __name__ == '__main__':
    main()
