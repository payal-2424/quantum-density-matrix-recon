import numpy as np
import torch
from torch.utils.data import Dataset

# Pauli labels: 0=I,1=X,2=Y,3=Z
PAULI_MATS = {
    0: np.array([[1,0],[0,1]], dtype=np.complex64),
    1: np.array([[0,1],[1,0]], dtype=np.complex64),
    2: np.array([[0,-1j],[1j,0]], dtype=np.complex64),
    3: np.array([[1,0],[0,-1]], dtype=np.complex64),
}

def kron_n(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def random_density_matrix(n_qubits: int, rank: int = None, seed=None):
    """
    Create a random PSD, trace-1 density matrix.
    If rank is provided, generate low-rank by truncation.
    """
    if seed is not None:
        np.random.seed(seed)
    dim = 2 ** n_qubits
    a = (np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)).astype(np.complex64)
    rho = a @ a.conj().T
    if rank is not None and rank < dim:
        # project to low rank via eigen-truncation
        w, v = np.linalg.eigh(rho)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:, idx]
        w[rank:] = 0
        rho = (v * w) @ v.conj().T
    rho = rho / np.trace(rho)
    return rho.astype(np.complex64)

def sample_pauli_string(n_qubits: int):
    # each qubit measured in {X,Y,Z} (avoid I to match "measurement basis" style)
    # map X=1, Y=2, Z=3
    return np.random.randint(1, 4, size=(n_qubits,), dtype=np.int64)

def pauli_operator(pauli_string):
    mats = [PAULI_MATS[int(p)] for p in pauli_string]
    return kron_n(mats)

def measure_expectation(rho, pauli_string, noise_std: float = 0.0):
    P = pauli_operator(pauli_string)
    val = np.trace(rho @ P).real  # expectation is real for Hermitian rho, Pauli operator
    if noise_std > 0:
        val += np.random.randn() * noise_std
    # clip into physical range for Pauli expectations
    return float(np.clip(val, -1.0, 1.0))

def encode_record(pauli_string, exp_val):
    """
    Turn one measurement record into a token-like vector:
    - one-hot basis per qubit (X,Y,Z) => 3*n_qubits
    - append expectation value => +1
    Total: 3*n_qubits + 1
    """
    n = len(pauli_string)
    one_hot = np.zeros((n, 3), dtype=np.float32)
    for i, p in enumerate(pauli_string):
        # X=1->0, Y=2->1, Z=3->2
        one_hot[i, int(p)-1] = 1.0
    feat = np.concatenate([one_hot.reshape(-1), np.array([exp_val], dtype=np.float32)], axis=0)
    return feat

class ShadowDataset(Dataset):
    """
    Each sample:
      x: (T, D_in) sequence of T measurement records
      y: target density matrix rho (complex dim x dim)
    """
    def __init__(self, n_samples: int, n_qubits: int, T: int = 64, noise_std: float = 0.02, rank=None):
        self.n_samples = n_samples
        self.n_qubits = n_qubits
        self.T = T
        self.noise_std = noise_std
        self.rank = rank
        self.dim = 2 ** n_qubits
        self.d_in = 3*n_qubits + 1

        self.rhos = []
        self.X = []

        for _ in range(n_samples):
            rho = random_density_matrix(n_qubits, rank=rank)
            seq = []
            for _t in range(T):
                ps = sample_pauli_string(n_qubits)
                ev = measure_expectation(rho, ps, noise_std=noise_std)
                seq.append(encode_record(ps, ev))
            self.rhos.append(rho)
            self.X.append(np.stack(seq, axis=0))

        self.X = np.stack(self.X, axis=0).astype(np.float32)  # (N,T,D_in)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # float32
        rho = self.rhos[idx]
        # represent complex target as 2-channel real tensor
        rho_t = torch.from_numpy(np.stack([rho.real, rho.imag], axis=0))  # (2,dim,dim)
        return x, rho_t
