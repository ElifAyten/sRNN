# train_srnn.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import yaml

from srnn_helper import TrainConfig, fit_srnn_with_split

def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def cfg_from_yaml(d: dict) -> TrainConfig:
    exp   = d.get("experiment", {})
    data  = d.get("data", {})
    dr    = d.get("dr", {})
    model = d.get("model", {})
    train = d.get("training", {})
    init  = d.get("initialization", {})

    return TrainConfig(
        rat_id=int(data.get("rat_id", 0)),
        data_root=Path(data.get("data_root", "")) if data.get("data_root") else Path(),
        outputs_root=Path(data.get("outputs_root", "")) if data.get("outputs_root") else Path(),
        subset_name=str(data.get("subset", "responsive")),
        dr_method=str(dr.get("method", "dca1")),
        dr_n=int(dr.get("n_components", 8)),
        dr_random_state=dr.get("random_state", exp.get("seed", 0)),
        K_states=int(model.get("K_states", 5)),
        latent_dim=int(model.get("latent_dim", 8)),
        kappa=float(model.get("kappa", 0.0)),
        num_iters=int(train.get("num_iters", 2000)),
        warmup_epochs=int(init.get("warmup_epochs", 200)),
        window_size=int(train.get("window_size", 100)),
        stride=int(train.get("stride", 1)),
        batch_size=int(train.get("batch_size", 128)),
        lr=float(train.get("lr", 1e-4)),
        seed=int(exp.get("seed", 0)),
        test_split=float(train.get("test_split", 0.2)),
        overwrite=bool(train.get("overwrite", False)),
        verbose=bool(train.get("verbose", True)),
        lambda_entropy=float(init.get("lambda_entropy", 1e-3)),
        lambda_usage=float(init.get("lambda_usage", 1e-2)),
        ms_per_sample=train.get("ms_per_sample", None),
        rate_mode=str(train.get("rate_mode", "mean")),
    )

def main():
    ap = argparse.ArgumentParser(description="Train SRNN from a YAML config.")
    ap.add_argument("--config", "-c", required=True, help="Path to YAML config.")
    ap.add_argument("--rats", type=int, nargs="*", help="Override: list of rat IDs to run.")
    ap.add_argument("--kappa-grid", type=float, nargs="*", help="Override: sweep κ values.")
    args = ap.parse_args()

    conf = load_yaml(args.config)

    rats = args.rats or conf.get("data", {}).get("rats", None)
    if rats is None:
        rats = [conf.get("data", {}).get("rat_id", 0)]
    if not isinstance(rats, (list, tuple)):
        rats = [rats]

    kappa_grid = args.kappa_grid or conf.get("model", {}).get("kappa_values", None)
    if kappa_grid is None:
        kappa_grid = [conf.get("model", {}).get("kappa", 0.0)]

    results = []
    for rid in rats:
        conf_mod = dict(conf)  # shallow copy
        conf_mod.setdefault("data", {}).update({"rat_id": rid})
        for kappa in kappa_grid:
            conf_mod.setdefault("model", {})["kappa"] = float(kappa)
            cfg = cfg_from_yaml(conf_mod)
            out = fit_srnn_with_split(cfg)
            results.append({"rat": rid, "kappa": kappa, **out})

    # pretty print + save a small summary next to the config
    print(json.dumps(results, indent=2))
    out_path = Path(args.config).with_suffix(".results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved summary → {out_path}")

if __name__ == "__main__":
    sys.exit(main())

