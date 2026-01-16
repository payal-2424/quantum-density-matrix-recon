import torch

def to_complex(rho_2ch):
    # rho_2ch: (B,2,d,d) or (2,d,d)
    if rho_2ch.dim() == 4:
        return torch.complex(rho_2ch[:,0], rho_2ch[:,1])
    return torch.complex(rho_2ch[0], rho_2ch[1])

def trace_distance(rho, sigma):
    """
    Trace distance: 0.5 * ||rho - sigma||_1
    Compute via eigenvalues of Hermitian (rho-sigma).
    """
    d = rho - sigma
    dH = 0.5 * (d + torch.conj(d.transpose(-1, -2)))
    evals = torch.linalg.eigvalsh(dH)  # real
    return 0.5 * torch.sum(torch.abs(evals), dim=-1)


def fidelity(rho, sigma):
    """
    Uhlmann fidelity: F(r,s) = (Tr sqrt(sqrt(r) s sqrt(r)))^2
    Batched version.
    """
    # Hermitian symmetrize for numerical stability
    rhoH = 0.5 * (rho + torch.conj(rho.transpose(-1, -2)))
    sigmaH = 0.5 * (sigma + torch.conj(sigma.transpose(-1, -2)))

    # eigendecomposition of rho
    w, v = torch.linalg.eigh(rhoH)          # w: (B,d), v: (B,d,d)
    w = torch.clamp(w.real, min=0.0)

    # sqrt(rho) = V diag(sqrt(w)) V†
    sqrt_w = torch.sqrt(w).unsqueeze(-2)    # (B,1,d) so it scales columns of V
    sqrt_rho = (v * sqrt_w) @ torch.conj(v.transpose(-1, -2))

    # mid = sqrt(rho) sigma sqrt(rho)
    mid = sqrt_rho @ sigmaH @ sqrt_rho
    midH = 0.5 * (mid + torch.conj(mid.transpose(-1, -2)))

    w2, _ = torch.linalg.eigh(midH)
    w2 = torch.clamp(w2.real, min=0.0)

    tr_sqrt = torch.sum(torch.sqrt(w2), dim=-1)
    return torch.clamp(tr_sqrt**2, 0.0, 1.0)

