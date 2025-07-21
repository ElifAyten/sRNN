# rSLDS/create_sub_data.py
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

def _macro_area(area: str) -> str | None:
    """Map fine labels to coarse macro‐areas."""
    a = area.lower()
    if a.startswith("d"): return "dorsal"
    if a.startswith("v"): return "ventral"
    if a == "thalamus":   return "thalamus"
    return None

def export_area_splits(
    hdf5_path   : str | Path,
    units_csv   : str | Path,
    out_dir     : str | Path,
    rat_tag     : str,
    min_spikes  : int = 1,
    verbose     : bool = True
) -> Dict[str, List[str]]:
    """
    Create wide+long CSVs for dorsal / ventral / thalamus populations.
    - hdf5_path: full path to the .hdf5 file
    - units_csv: master units.csv
    - out_dir:   directory to write your *_wide.csv and *_long.csv
    - rat_tag:   e.g. "Rat3" or "Rat15"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) read the metadata for just this rat
    units = (
        pd.read_csv(units_csv)
          .query("rat == @rat_tag")
          .assign(
            hdf5_key=lambda df: "cluster" + df["cluster"].astype(str),
            macro    = lambda df: df["area"].map(_macro_area)
          )
          .dropna(subset=["macro"])
    )
    if units.empty:
        raise ValueError(f"No rows for {rat_tag} in {units_csv}")

    # 2) pull everything from the HDF5 *inside* the with‐block
    with h5py.File(hdf5_path, "r") as f:
        time_vec   = f["time"][...]
        speed      = f["speed"][...]
        rates_all  = f["firing_rates"][...]    # shape (n_units × T)
        spike_grp  = f["spike_times"]
        raw_keys   = list(spike_grp.keys())    # e.g. "cluster1031_0", …

        # build map: exact raw_key → row index in rates_all
        key2idx    = {k: i for i, k in enumerate(raw_keys)}
        spike_counts = np.array([ spike_grp[k].size for k in raw_keys ])
        is_active    = spike_counts >= min_spikes

    # 3) keep only those metadata rows whose hdf5_key matches one of raw_keys
    units = (
      units[units["hdf5_key"].isin(raw_keys)]
           .assign(row_idx=lambda df: df["hdf5_key"].map(key2idx))
    )
    # mask by activity
    units = units[units["row_idx"].notna() & is_active[units["row_idx"].astype(int)]]
    units["row_idx"] = units["row_idx"].astype(int)

    if verbose:
        print(f"{rat_tag}: active units ≥{min_spikes} spikes → {len(units)}")

    exported: Dict[str, List[str]] = {}
    for macro in sorted(units["macro"].unique()):
        sub = units[units["macro"] == macro]
        if sub.empty:
            if verbose: print(f"[{macro}] skipped (no units)")
            continue
        if verbose: print(f"[{macro}] {len(sub)} neurons → exporting")

        # wide-format CSV
        wide = pd.DataFrame(rates_all[sub.row_idx].T,
                            columns=sub["cluster"].astype(str))
        wide.insert(0, "time_s", time_vec)
        wide["speed"] = speed
        wide_path = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_path, index=False)

        # long-format CSV
        recs = []
        for _, row in sub.iterrows():
            rates = rates_all[row.row_idx]
            recs.extend({
                "time_s"     : t,
                "speed"      : s,
                "cluster"    : row.cluster,
                "neuron_type": row.neuron_type,
                "area"       : row.area,
                "macro_area" : macro,
                "rate"       : r
            } for t, s, r in zip(time_vec, speed, rates))

        long_path = out_dir / f"{macro}_long.csv"
        pd.DataFrame.from_records(recs).to_csv(long_path, index=False)

        exported[macro] = [str(wide_path), str(long_path)]
        if verbose:
            print(f"[{macro}] wrote {wide_path.name} + {long_path.name}")

    if verbose:
        print(f"✓ All splits saved in: {out_dir}/")

    return exported
