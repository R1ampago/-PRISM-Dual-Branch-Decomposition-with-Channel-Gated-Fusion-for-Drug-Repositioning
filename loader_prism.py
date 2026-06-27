"""
Lightweight data loader for PRISM — supports .mat, CSV, and TSV formats.
"""
import numpy as np, scipy.sparse as sp, scipy.io as scio
import pandas as pd, os
from sklearn.model_selection import KFold

class PRISMLoader:
    def __init__(self, name):
        data_dir = f'./drug_data/{name}'
        if name == 'Ldataset':
            drug_sim = pd.read_csv(os.path.join(data_dir, 'drug_sim.csv'),
                                   header=0, index_col=0).values.astype(np.float32)
            dis_sim  = pd.read_csv(os.path.join(data_dir, 'dis_sim.csv'),
                                   header=0, index_col=0).values.astype(np.float32)
            drug_dis = pd.read_csv(os.path.join(data_dir, 'drug_dis.csv'),
                                   header=0, index_col=0).values.astype(np.int8)
            self.A = drug_dis  # [N_drug, N_dis]
            self.drug_sim = drug_sim
            self.dis_sim = dis_sim
        elif name == 'LRSSL':
            drug_sim = pd.read_csv(os.path.join(data_dir, 'drug_sim.txt'),
                                   sep='\t', header=0, index_col=0).values.astype(np.float32)
            dis_sim  = pd.read_csv(os.path.join(data_dir, 'dis_sim.txt'),
                                   sep='\t', header=0, index_col=0).values.astype(np.float32)
            drug_dis = pd.read_csv(os.path.join(data_dir, 'drug_dis.txt'),
                                   sep='\t', header=0, index_col=0).values.astype(np.int8)
            self.A = drug_dis  # [N_drug, N_dis]
            self.drug_sim = drug_sim
            self.dis_sim = dis_sim
        else:
            d = scio.loadmat(os.path.join(data_dir, f'{name}.mat'))
            self.A = d['didr'].T  # [N_drug, N_dis]
            self.drug_sim = d['drug']
            self.dis_sim = d['disease']

        self.n_drug, self.n_dis = self.A.shape
        print(f"  {self.n_drug}x{self.n_dis}, density={self.A.mean()*100:.2f}%")

        # Pre-compute 10-fold CV splits
        kf = KFold(n_splits=10, shuffle=True, random_state=1024)
        pos_r, pos_c = np.nonzero(self.A)
        neg_r, neg_c = np.nonzero(1 - self.A)
        self.pos_splits = list(kf.split(pos_r))
        self.neg_splits = list(kf.split(neg_r))
        self.pos_r, self.pos_c = pos_r, pos_c
        self.neg_r, self.neg_c = neg_r, neg_c

    def get_fold(self, cv_idx):
        train_pos_idx, test_pos_idx = self.pos_splits[cv_idx]
        train_neg_idx, test_neg_idx = self.neg_splits[cv_idx]

        pr = self.pos_r[train_pos_idx]; pd = self.pos_c[train_pos_idx]; n_pos = len(pr)
        nr = self.neg_r[train_neg_idx]; nd = self.neg_c[train_neg_idx]
        tr = self.pos_r[test_pos_idx];  td = self.pos_c[test_pos_idx]
        tnr = self.neg_r[test_neg_idx]; tnd = self.neg_c[test_neg_idx]

        test_r = np.concatenate([tr, tnr])
        test_d = np.concatenate([td, tnd])
        test_l = np.concatenate([np.ones(len(tr)), np.zeros(len(tnr))])

        tam = np.zeros_like(self.A)
        tam[pr, pd] = 1
        A_train = sp.csr_matrix(tam)
        e = np.array(A_train.nonzero()).T
        return e, (test_r, test_d, test_l), (pr, pd, n_pos, nr, nd), A_train
