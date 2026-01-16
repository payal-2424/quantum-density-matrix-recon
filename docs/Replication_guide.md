# Replication Guide

## 1) Environment setup
Recommended: Python 3.10+

Install dependencies:
pip install torch numpy

(If you want GPU, install a CUDA-enabled PyTorch build from the official PyTorch instructions.)

## 2) Repo structure
- src/ contains all code
- outputs/ stores checkpoints and results
- docs/ contains explanation and replication steps

## 3) Generate dataset (done inside training script)
The dataset is generated on-the-fly in `src/data.py`:
- Random density matrices ρ are created
- Pauli-string measurements are sampled
- Expectation values are computed (optional Gaussian noise)

No separate dataset file is required.

## 4) Train
From repo root:
python src/train.py --n_qubits 2 --T 64 --train_samples 4000 --test_samples 800 --epochs 25

Outputs:
- outputs/model.pt (best checkpoint by test fidelity)
- outputs/train_log.json (epoch-wise logs)

## 5) Evaluate + required metrics
python src/eval.py --ckpt outputs/model.pt --n_qubits 2 --T 64 --test_samples 800

Outputs:
- outputs/metrics_test.json including:
  - mean fidelity
  - mean trace distance
  - inference latency (sec/sample)

## 6) Reproducing exact results
Set the same seed:
python src/train.py --seed 42 ...
