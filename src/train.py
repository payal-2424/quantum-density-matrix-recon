import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data import ShadowDataset
from model import ShadowTransformer
from metrics import to_complex, fidelity, trace_distance
from utils import set_seed, ensure_dir, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, default=2)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--train_samples", type=int, default=4000)
    ap.add_argument("--test_samples", type=int, default=800)
    ap.add_argument("--noise_std", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2 ** args.n_qubits
    d_in = 3*args.n_qubits + 1

    full = ShadowDataset(
        n_samples=args.train_samples + args.test_samples,
        n_qubits=args.n_qubits,
        T=args.T,
        noise_std=args.noise_std
    )
    train_len = args.train_samples
    test_len = args.test_samples
    train_ds, test_ds = random_split(full, [train_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ShadowTransformer(d_in=d_in, n_qubits=args.n_qubits).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_fid = -1.0
    log = {"args": vars(args), "epochs": []}

    for ep in range(1, args.epochs+1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # evaluate
        model.eval()
        fids = []
        tds = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)

                rho = to_complex(pred)
                sigma = to_complex(y)

                f = fidelity(rho, sigma)
                td = trace_distance(rho, sigma)

                fids.append(f.detach().cpu())
                tds.append(td.detach().cpu())

        mean_fid = torch.cat(fids).mean().item()
        mean_td = torch.cat(tds).mean().item()

        log["epochs"].append({
            "epoch": ep,
            "train_loss": train_loss,
            "mean_fidelity": mean_fid,
            "mean_trace_distance": mean_td
        })

        # save best
        if mean_fid > best_fid:
            best_fid = mean_fid
            torch.save({
                "model_state": model.state_dict(),
                "n_qubits": args.n_qubits,
                "T": args.T,
                "d_in": d_in
            }, f"{args.outdir}/model.pt")

        print(f"Epoch {ep:02d} | loss {train_loss:.5f} | fid {mean_fid:.5f} | td {mean_td:.5f}")

    save_json(f"{args.outdir}/train_log.json", log)

if __name__ == "__main__":
    main()
