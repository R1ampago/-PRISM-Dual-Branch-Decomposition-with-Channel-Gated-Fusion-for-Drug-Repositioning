"""
PRISM full CV benchmark — 3 datasets × 3 seeds × 10 folds.
Checkpoint selection: Comb = 0.8*AUC + 0.2*AUPR.
All hyperparameters match main.py.
"""
import numpy as np, torch as th, pickle, os
from model import PRISM
from loader_prism import PRISMLoader
from train import train_fold

DEV = 'cuda:0' if th.cuda.is_available() else 'cpu'
EPOCHS = 3000; BS = 8192; EVAL_EVERY = 20; NEG_RATIO = 10

DATASETS = ['Fdataset', 'Cdataset', 'Gdataset']
SEEDS = [77, 42, 123]

os.makedirs('results', exist_ok=True)

all_results = {}  # {(ds, seed): (auc_mean, auc_std, aupr_mean, aupr_std)}

for ds in DATASETS:
    loader = PRISMLoader(ds)
    ndr, n_dis = loader.n_drug, loader.n_dis

    cache_adj = f'cache/cache_{ds}_adj.pkl'
    with open(cache_adj, 'rb') as f:
        pre = pickle.load(f)
    drug_adjs, dis_adjs = pre['adjs_d'], pre['adjs_p']

    for seed in SEEDS:
        key = f'{ds}_s{seed}'
        print(f'\n{"=" * 60}')
        print(f'  {key}')
        print(f'{"=" * 60}')

        th.manual_seed(seed); np.random.seed(seed)

        fold_auc, fold_aupr = [], []

        for fold in range(10):
            edges, (test_r, test_d, test_l), (pr, pd, n_pos, nr, nd), _ = loader.get_fold(fold)
            ei = th.stack([th.LongTensor(edges[:, 0]).to(DEV),
                           th.LongTensor(edges[:, 1]).to(DEV)])

            model = PRISM(ndr, n_dis, drug_adjs, dis_adjs,
                          hidden_dim=256, embed_dim=128,
                          dropout=0.3, ch_dropout=0.4).to(DEV)

            fold_data = (ei, test_r, test_d, test_l, pr, pd, n_pos, nr, nd)
            save_dir = f'cache/fullcv/{key}'
            os.makedirs(save_dir, exist_ok=True)

            auc, aupr, ep = train_fold(
                model, fold_data, epochs=EPOCHS,
                batch_size=BS, feat_lr=0.004, rest_lr=0.002,
                weight_decay=1e-5, eval_every=EVAL_EVERY,
                selection='comb', device=DEV, neg_ratio=NEG_RATIO,
                save_path=f'{save_dir}/fold{fold}.pth')

            fold_auc.append(auc); fold_aupr.append(aupr)
            print(f'    F{fold}: AUC={auc:.4f} AUPR={aupr:.4f} @E{ep}', flush=True)

        auc_m, auc_s = np.mean(fold_auc), np.std(fold_auc)
        aupr_m, aupr_s = np.mean(fold_aupr), np.std(fold_aupr)
        print(f'  [{key}] AUC={auc_m:.4f}±{auc_s:.4f}  AUPR={aupr_m:.4f}±{aupr_s:.4f}')
        all_results[key] = (auc_m, auc_s, aupr_m, aupr_s)

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print(f'\n{"=" * 65}')
print('  FINAL SUMMARY')
print(f'{"=" * 65}')
print(f'  {"Dataset":<12s} {"AUC":>18s} {"AUPR":>18s}')
print(f'  {"─" * 50}')

for ds in DATASETS:
    aucs = [all_results[f'{ds}_s{s}'][0] for s in SEEDS]
    auprs = [all_results[f'{ds}_s{s}'][2] for s in SEEDS]
    print(f'  {ds:<12s} {np.mean(aucs):.4f}±{np.std(aucs):.4f}     '
          f'{np.mean(auprs):.4f}±{np.std(auprs):.4f}')

print(f'\n  {"Per-seed":─^50}')
for ds in DATASETS:
    for s in SEEDS:
        auc_m, auc_s, aupr_m, aupr_s = all_results[f'{ds}_s{s}']
        print(f'  {ds:<12s} s{s:<4d} {auc_m:.4f}±{auc_s:.4f}       {aupr_m:.4f}±{aupr_s:.4f}')

# Save
with open('results/full_cv_results.txt', 'w') as f:
    f.write('PRISM Full CV Results (Comb Selection)\n')
    f.write('=' * 42 + '\n\n')
    for ds in DATASETS:
        aucs = [all_results[f'{ds}_s{s}'][0] for s in SEEDS]
        auprs = [all_results[f'{ds}_s{s}'][2] for s in SEEDS]
        f.write(f'{ds}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  '
                f'AUPR={np.mean(auprs):.4f}±{np.std(auprs):.4f}\n')
    f.write('\nPer-seed:\n')
    for key, (auc_m, auc_s, aupr_m, aupr_s) in all_results.items():
        f.write(f'  {key}: AUC={auc_m:.4f}±{auc_s:.4f} AUPR={aupr_m:.4f}±{aupr_s:.4f}\n')

print(f'\n  Results saved to results/full_cv_results.txt')
