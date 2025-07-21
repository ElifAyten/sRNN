# src/dataset.py
import os, h5py, numpy as np
from pathlib import Path

def _auto_h5_path(folder: Path, prefix="NpxFiringRate") -> Path:
    """Pick the first *.hdf5* file whose name starts with `prefix`."""
    for f in folder.iterdir():
        if f.is_file() and f.name.startswith(prefix) and f.suffix in {".h5", ".hdf5"}:
            return f
    raise FileNotFoundError(f"No file starting with '{prefix}' in {folder}")

def _optional_dataset(h5, key_list):
    """Return the first dataset found among the candidate keys, else None."""
    for k in key_list:
        if "/" in k:                      # nested, e.g. "behavior/speed"
            grp, ds = k.split("/", 1)
            if grp in h5 and ds in h5[grp]:
                return np.asarray(h5[grp][ds])
        elif k in h5:                     # top-level dataset
            return np.asarray(h5[k])
    return None

# ----------------------------------------------------------------------
def load_rat_data(base_dir, rat_id, *, prefix="NpxFiringRate", verbose=True):
    """
    base_dir : str | Path      directory that contains "RatXX/" sub-folders
    rat_id   : int | str       15  ➔ "Rat15"
    Returns dict: time, firing_rates, speed, footshock_times, pupil (may be None)
    """
    folder = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_path = _auto_h5_path(folder, prefix=prefix)

    with h5py.File(h5_path, "r") as hdf:
        firing_rates   = np.asarray(hdf["firing_rates"]).T     # (time, neurons)
        time           = np.asarray(hdf["time"])
        footshock_times= np.asarray(hdf["footshock_times"])
        speed          = _optional_dataset(hdf, ["speed", "behavior/speed"])
        pupil          = _optional_dataset(hdf, ["pupil_diameter",
                                                 "behavior/pupil_diameter",
                                                 "pupil"])

        spike_keys = list(hdf["spike_times"].keys()) if "spike_times" in hdf else None

    if verbose:
        print(f"\nLoaded: {h5_path}")
        print("  firing_rates:", firing_rates.shape)
        print("  time        :", time.shape)
        print("  footshocks  :", footshock_times.shape)
        print("  speed       :", None if speed is None else speed.shape)
        print("  pupil       :", None if pupil is None else pupil.shape)
        if spike_keys: print("  spike_times keys:", spike_keys)

    return dict(time=time,
                firing_rates=firing_rates,
                speed=speed,
                footshock_times=footshock_times,
                pupil=pupil)
