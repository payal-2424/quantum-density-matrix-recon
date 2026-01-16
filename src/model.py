import torch
import torch.nn as nn
import torch.nn.functional as F

def build_lower_triangular(vec, dim):
    """
    vec: (B, K) real values for lower triangle (including diag)
    returns L: (B, dim, dim) real
    """
    B = vec.shape[0]
    L = torch.zeros((B, dim, dim), device=vec.device, dtype=vec.dtype)
    idx = 0
    for i in range(dim):
        for j in range(i+1):
            L[:, i, j] = vec[:, idx]
            idx += 1
    return L

class ShadowTransformer(nn.Module):
    """
    Input: sequence (B,T,D_in)
    Output: parameters of complex lower-triangular L (real+imag)
    """
    def __init__(self, d_in, n_qubits, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.d_in = d_in
        self.d_model = d_model

        self.input_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # pool -> output head
        self.pool = nn.Linear(d_model, d_model)

        # number of lower-tri entries (including diag)
        self.k = self.dim * (self.dim + 1) // 2

        # output real and imag parts
        self.head_real = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.k),
        )
        self.head_imag = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.k),
        )

    def forward(self, x):
        """
        x: (B,T,D_in)
        returns rho_pred: (B,2,dim,dim) (real/imag)
        """
        h = self.input_proj(x)               # (B,T,d_model)
        h = self.encoder(h)                  # (B,T,d_model)
        h = h.mean(dim=1)                    # mean pool (B,d_model)
        h = torch.tanh(self.pool(h))         # (B,d_model)

        v_r = self.head_real(h)              # (B,k)
        v_i = self.head_imag(h)              # (B,k)

        # build lower triangle
        Lr = build_lower_triangular(v_r, self.dim)
        Li = build_lower_triangular(v_i, self.dim)

        # enforce positive diagonal on real part; imag diag should be 0 for stability
        diag = torch.diagonal(Lr, dim1=-2, dim2=-1)
        diag_pos = F.softplus(diag) + 1e-6
        Lr = Lr.clone()
        Lr.diagonal(dim1=-2, dim2=-1).copy_(diag_pos)
        Li = Li.clone()
        Li.diagonal(dim1=-2, dim2=-1).zero_()

        # complex L
        L = torch.complex(Lr, Li)            # (B,dim,dim)

        # rho = L L^\dag / Tr
        rho = L @ torch.conj(L.transpose(-1, -2))  # (B,dim,dim)
        tr = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)).clamp_min(1e-8)
        rho = rho / tr.view(-1, 1, 1)

        rho_out = torch.stack([rho.real, rho.imag], dim=1)  # (B,2,dim,dim)
        return rho_out
