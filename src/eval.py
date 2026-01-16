import argparse
import torch
from torch.utils.data import DataLoader

from data import ShadowDataset
from model import ShadowTransformer
from metrics import to_complex, fidelity, trace_distance
from utils import Timer, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/model.pt")
    ap.add_argument("--n_qubits", type=int, default=2)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--test_samples", type=int, default=800)
    ap.add_argument("--noise_std", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", type=str, default="outputs/metrics_test.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_in = 3*args.n_qubits + 1

    ds = ShadowDataset(
        n_samples=args.test_samples,
        n_qubits=args.n_qubits,
        T=args.T,
        noise_std=args.noise_std
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = ShadowTransformer(d_in=d_in, n_qubits=args.n_qubits).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    fids = []
    tds = []
    latencies = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            with Timer() as t:
                pred = model(x)
            # latency per sample
            latencies.append(t.dt / x.shape[0])

            rho = to_complex(pred)
            sigma = to_complex(y)
            fids.append(fidelity(rho, sigma).cpu())
            tds.append(trace_distance(rho, sigma).cpu())

    fids = torch.cat(fids)
    tds = torch.cat(tds)

    out = {
        "mean_fidelity": float(fids.mean().item()),
        "mean_trace_distance": float(tds.mean().item()),
        "inference_latency_sec_per_sample": float(sum(latencies)/len(latencies)),
        "n_qubits": args.n_qubits,
        "T": args.T,
        "test_samples": args.test_samples
    }
    save_json(args.out, out)
    print(out)

if __name__ == "__main__":
    main()
