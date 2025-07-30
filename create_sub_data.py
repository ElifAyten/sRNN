"""sRNN/create_sub_data.py

A patched version of `export_area_splits` that lets you filter **in‑helper** by
(1) foot‑shock responsiveness (`shocks_response` ∈ {"excited","inhibited", …}) and
(2) minimum mean firing‑rate (Hz).

Drop this file in place of the original or import it under a new name; the
call signature in your notebook stays almost identical to the old one.

Example
-------
>>> from create_sub_data import export_area_splits
>>> export_area_splits(
...     hdf5_path   = "Rat4/...hdf5",
...     units_csv   = "units.csv",
...     out_dir     = "rat4_area_resp0p5Hz_10ms",
...     rat_tag     = "Rat4",
...     min_spikes  = 1,
...     min_rate_hz = 0.5,                         # NEW (default 0.0)
...     responses   = ("excited", "inhibited"),     # NEW (default None → keep all)
... )
"""

from __future__ import annotations

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

__all__ = ["export_area_splits"]

# -----------------------------------------------------------------------------
# Helper to collapse fine anatomical labels → macro‑areas
# -----------------------------------------------------------------------------

def _macro_area(area: str) -> str | None:
    a = area.lower()
    if a.startswith("d"):
        return "dorsal"
    if a.startswith("v"):
        return "ventral"
    if a == "thalamus":
        return "thalamus"
    return None  # anything else gets dropped

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def export_area_splits(
    hdf5_path: str | Path,
    units_csv: str | Path,
    out_dir: str | Path,
    rat_tag: str,
    *,
    min_spikes: int = 1,
    min_rate_hz: float = 0.0,
    responses: Iterable[str] | None = None,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Export wide and long CSV tables for each macro‑area.

    Parameters
    ----------
    hdf5_path : str or Path
        Full path to the session HDF5 file.
    units_csv : str or Path
        The master *units.csv* with per‑cluster metadata.
    out_dir : str or Path
        Destination folder for the generated CSVs.
    rat_tag : str
        e.g. "Rat4". Used to subset *units.csv* rows.
    min_spikes : int, default 1
        Keep units that fired at least this many spikes in the entire session.
    min_rate_hz : float, default 0.0
        Keep units whose **mean firing‑rate ≥ this threshold** (after spike mask).
    responses : Iterable[str] or None, default None
        If given, keep only rows whose *shocks_response* column is in this list.
        Example: ("excited", "inhibited"). Pass *None* to disable.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    dict {macro_area → [wide_csv_path, long_csv_path]}
    """

    hdf5_path = Path(hdf5_path).expanduser()
    units_csv = Path(units_csv).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Metadata rows for this rat
    # ------------------------------------------------------------------
    units = (
        pd.read_csv(units_csv)
        .query("rat == @rat_tag")
        .assign(
            hdf5_key=lambda df: "cluster" + df["cluster"].astype(str),
            macro=lambda df: df["area"].map(_macro_area),
        )
        .dropna(subset=["macro"])
    )
    if units.empty:
        raise ValueError(f"No rows for {rat_tag} in {units_csv}")

    # optional: keep only specified shock‑response classes
    if responses is not None:
        responses_set = set(responses)
        units = units[units["shocks_response"].isin(responses_set)]
        if verbose:
            print(
                f"{rat_tag}: filter by shocks_response ∈ {responses_set} → {len(units)} rows"
            )
        if units.empty:
            raise ValueError("No units left after shocks_response filter.")

    # ------------------------------------------------------------------
    # 2) Pull arrays from HDF5
    # ------------------------------------------------------------------
    with h5py.File(hdf5_path, "r") as f:
        time_vec = f["time"][...]
        dt = time_vec[1] - time_vec[0]
        speed = f["speed"][...]
        rates_all = f["firing_rates"][...]  # shape (n_units × T)
        spike_grp = f["spike_times"]
        raw_keys = list(spike_grp.keys())  # "cluster1031_0", …

        # Map raw_key → row index in rates_all
        key2idx = {k: i for i, k in enumerate(raw_keys)}
        spike_counts = np.array([spike_grp[k].size for k in raw_keys])
        is_active = spike_counts >= min_spikes

    # ------------------------------------------------------------------
    # 3) Cross‑match metadata ↔ HDF5 rows & basic activity filter
    # ------------------------------------------------------------------
    units = (
        units[units["hdf5_key"].isin(raw_keys)]
        .assign(row_idx=lambda df: df["hdf5_key"].map(key2idx).astype(int))
    )
    units = units[is_active[units["row_idx"]]]
    if verbose:
        print(f"{rat_tag}: ≥{min_spikes} spikes → {len(units)} units after match")
    if units.empty:
        raise ValueError("No units left after min_spikes filter.")

    # ------------------------------------------------------------------
    # 4) Mean‑rate (Hz) filter
    # ------------------------------------------------------------------
    mean_rate_hz = rates_all[units.row_idx].mean(1) / dt  # spikes / dt → Hz
    units = units.assign(mean_rate_hz=mean_rate_hz)
    units = units[units.mean_rate_hz >= min_rate_hz]
    if verbose:
        print(
            f"{rat_tag}: ≥{min_rate_hz} Hz → {len(units)} units after rate filter"
        )
    if units.empty:
        raise ValueError("No units left after mean‑rate filter.")

    # ------------------------------------------------------------------
    # 5) Export per‑area CSVs
    # ------------------------------------------------------------------
    exported: Dict[str, List[str]] = {}
    for macro in sorted(units["macro"].unique()):
        sub = units[units["macro"] == macro]
        if sub.empty:
            continue

        if verbose:
            print(f"[{macro}] {len(sub)} neurons → exporting")

        # ---------------- wide format ----------------
        wide = pd.DataFrame(rates_all[sub.row_idx].T, columns=sub["cluster"].astype(str))
        wide.insert(0, "time_s", time_vec)
        wide["speed"] = speed
        wide_path = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_path, index=False)

        # ---------------- long format ----------------
        recs: list[dict] = []
        for _, row in sub.iterrows():
            rates = rates_all[row.row_idx]
            recs.extend(
                {
                    "time_s": t,
                    "speed": s,
                    "cluster": row.cluster,
                    "neuron_type": row.neuron_type,
                    "area": row.area,
                    "macro_area": macro,
                    "rate": r,
                }
                for t, s, r in zip(time_vec, speed, rates)
            )
        long_path = out_dir / f"{macro}_long.csv"
        pd.DataFrame.from_records(recs).to_csv(long_path, index=False)

        exported[macro] = [str(wide_path), str(long_path)]
        if verbose:
            print(f"[{macro}] wrote {wide_path.name} + {long_path.name}")

    if verbose:
        print(f"✓ All splits saved in: {out_dir}/")

    return exported
