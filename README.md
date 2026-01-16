# quantum-density-matrix-recon

# Assignment 2 — Density Matrix Reconstruction (QCG × PaAC Open Project, Winter 2025–2026)

This project trains a neural model to reconstruct a valid density matrix ρ from measurement data while strictly enforcing:
- Hermitian
- Positive Semi-Definite (PSD)
- Unit trace

We implement Track 1 (Classical Shadows + Transformer) and enforce physical constraints using a Cholesky factorization:
ρ = (L L†) / Tr(L L†)

## Repo structure
- `src/` core code (data generation, model, training, evaluation)
- `outputs/` saved checkpoints + metric logs
- `docs/` detailed explanation + replication guide

## Quickstart
Install:
pip install torch numpy

Train:
python src/train.py --n_qubits 2 --T 64 --train_samples 4000 --test_samples 800 --epochs 25

Evaluate (required metrics + latency):
python src/eval.py --ckpt outputs/model.pt --n_qubits 2 --T 64 --test_samples 800

## Reported metrics (Part 4)
After running evaluation, see:
`outputs/metrics_test.json`
- Mean Fidelity across test set
- Mean Trace Distance across test set
- Inference latency (seconds per reconstruction)

## Documentation
- `docs/MODEL_WORKING.md`
- `docs/REPLICATION_GUIDE.md`

## AI attribution
See `AI_USAGE.md`
