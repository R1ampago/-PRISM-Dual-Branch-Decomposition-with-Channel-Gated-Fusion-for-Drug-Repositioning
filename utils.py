"""Utilities: KNN adjacency construction."""
import numpy as np
import scipy.sparse as sp


def knn_csr(sim, k):
    """Build KNN adjacency as sparse CSR matrix (normalized rows).

    Uses vectorized argpartition (O(n²)) instead of per-row argsort (O(n² log n)).
    """
    n = sim.shape[0]
    # Get top-(k+1) candidate indices per row via partial sort (k+1 accounts for self)
    cand = np.argpartition(-sim, k + 1, axis=1)[:, :k + 1]

    rows, cols, vals = [], [], []
    for i in range(n):
        # Exclude self from neighbors, then take top k
        neighbors = cand[i][cand[i] != i][:k]
        nodes = np.concatenate([[i], neighbors])
        w = 1.0 / len(nodes)
        for j in nodes:
            rows.append(i)
            cols.append(j)
            vals.append(w)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
