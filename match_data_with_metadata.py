# src/metadata.py
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

__all__ = ["match_units_to_hdf5"]

def _first_h5(base_dir: Path, rat_id: str | int, prefix="NpxFiringRate") -> Path:
    """Return the first HDF5 file in RatXX/ that starts with `prefix`."""
    rat_dir = base_dir / f"Rat{int(rat_id)}"
    for f in rat_dir.iterdir():
        if f.is_file() and f.name.startswith(prefix) and f.suffix in {".h5", ".hdf5"}:
            return f
    raise FileNotFoundError(f"No '{prefix}*.h5' under {rat_dir}")

def match_units_to_hdf5(units_csv: str | Path,
                        base_dir  : str | Path,
                        rat_id    : int | str,
                        *,
                        h5_prefix="NpxFiringRate",
                        spike_group="spike_times",
                        verbose=True):
    """
    Return a DataFrame that maps each metadata row (units.csv) to the row index
    of its firing-rate trace in the HDF5 file.

    Parameters
    ----------
    units_csv : path to the master units.csv
    base_dir  : directory that contains RatXX sub-folders
    rat_id    : 15  → "Rat15"
    h5_prefix : filename prefix before ".h5"
    spike_group : group name inside HDF5 that holds cluster datasets
    """
    base_dir = Path(base_dir)
    units_df = pd.read_csv(units_csv)

    # keep only rows for this rat
    rat_tag = f"Rat{int(rat_id)}"
    meta = units_df[units_df["rat"] == rat_tag].copy()
    if meta.empty:
        raise ValueError(f"No rows for {rat_tag} in {units_csv}")

    # build the key names that appear in /spike_times/cluster<N>
    meta["hdf5_key"] = "cluster" + meta["cluster"].astype(str)

    # discover HDF5 path and spike keys
    h5_path = _first_h5(base_dir, rat_id, prefix=h5_prefix)
    with h5py.File(h5_path, "r") as h5:
        spike_keys = sorted(h5[spike_group].keys())

    mapping = pd.DataFrame({
        "hdf5_key"         : spike_keys,
        "firing_rate_index": np.arange(len(spike_keys))
    })

    merged = meta.merge(mapping, on="hdf5_key", how="inner")

    if verbose:
        print(f"Loaded metadata rows for {rat_tag}: {len(meta)}")
        print(f"Matched clusters in HDF5         : {len(merged)}")
        print(f"HDF5 file used: {h5_path}")

    return merged
