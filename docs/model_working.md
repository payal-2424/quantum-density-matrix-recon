# Model Working (Track 1: Classical-Shadows + Transformer)

## Goal
Given measurement data, reconstruct a valid density matrix ρ that satisfies:
- Hermitian
- Positive Semi-Definite (PSD)
- Unit trace

## Data representation (Classical-Shadow style)
Each training sample is a sequence of T measurement records.  
A record consists of:
1) Measurement basis per qubit (X/Y/Z encoded as one-hot)
2) A scalar expectation value ⟨P⟩ for the corresponding Pauli string operator P

So each token feature vector has size (3*n_qubits + 1), and the model input is a tensor of shape (T, D_in).

## Architecture
We use a Transformer Encoder:
- Linear projection: D_in → d_model
- L stacked encoder blocks (self-attention + MLP)
- Mean pooling over time to get a fixed-length representation

## Output parameterization with Cholesky factor (physical constraints)
The network outputs a lower-triangular matrix L (complex) via its flattened lower-triangle parameters.

Reconstruction:
ρ = (L L†) / Tr(L L†)

Why this enforces constraints:
- L L† is always Hermitian and PSD.
- Division by trace enforces Tr(ρ) = 1.

Implementation details:
- L is built as lower-triangular (including diagonal).
- Diagonal is forced positive using softplus to avoid degenerate solutions.
- Imaginary diagonal is set to 0 for stability.

## Loss and evaluation metrics
Training loss:
- MSE between predicted ρ and target ρ (real+imag channels)

Required evaluation metrics:
1) Quantum Fidelity F(ρ,σ) = (Tr sqrt( sqrt(ρ) σ sqrt(ρ) ))^2
2) Trace distance D(ρ,σ) = 1/2 ||ρ − σ||_1 computed via eigenvalues (Hermitian case)
