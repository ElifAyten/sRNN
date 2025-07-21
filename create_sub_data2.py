# rSLDS/create_responsive_tables.py
import h5py, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

__all__ = ["export_responsive_tables"]

def _macro_area(area: str) -> str | None:
    a = area.lower()
    if a.startswith("d"): return "dorsal"
    if a.startswith("v"): return "ventral"
    if a == "thalamus":   return "thalamus"
    return None                           # anything else → drop

def export_responsive_tables(
    hdf5_path   : str | Path,
    units_csv   : str | Path,
    out_dir     : str | Path,
    rat_tag     : str,                       # "Rat15" …
    *,
    responses   = ("excited", "inhibited"),  # which shock labels to keep
    verbose     = True,
):
    """
    Build **three CSV files** from one rat’s session, *only* for the neurons
    with `shocks_response` ∈ `responses`.

    1. `<out_dir>/responsive_rates_raw.csv`       – time_s + raw firing-rate/Hz
    2. `<out_dir>/responsive_rates_z.csv`         – z-scored version
    3. `<out_dir>/responsive_metadata.csv`        – metadata rows that match

    The function returns a dict with handy Numpy arrays that you can feed
    straight into modelling pipelines.

    Notes
    -----
    • Firing rates are re-binned from `spike_times` instead of using
      the pre-binned dataset inside the file.  
    • Speed and foot-shock vectors are returned (min-max & 0/1).
    """

    h5_path = Path(hdf5_path).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # metadata 
    units = (
        pd.read_csv(units_csv)
          .query("rat == @rat_tag")
          .assign(
              hdf5_key = lambda d: "cluster" + d["cluster"].astype(str),
              macro    = lambda d: d["area"].map(_macro_area)
          )
    )
    responsive = units[units["shocks_response"].isin(responses)].copy()
    if responsive.empty:
        raise ValueError(f"No rows with shocks_response in {responses}")

    # open the HDF5 once
    with h5py.File(h5_path, "r") as f:
        time_vec        = f["time"][...]
        dt              = time_vec[1] - time_vec[0]
        edges           = np.concatenate([time_vec - dt/2, [time_vec[-1] + dt/2]])
        spikes_grp      = f["spike_times"]
        speed           = f["speed"][...]
        footshock_times = f["footshock_times"][...]

        # Build key → row index map
        key2idx = {k: i for i, k in enumerate(spikes_grp.keys())}

        # Keep only keys that exist
        responsive = responsive[responsive["hdf5_key"].isin(spikes_grp.keys())]
        responsive["row_idx"] = responsive["hdf5_key"].map(key2idx).astype(int)

        if responsive.empty:
            raise ValueError("None of the responsive units are in spike_times.")

        # histogram & build raw-rate matrix
        raw_rate_df = pd.DataFrame({"time_s": time_vec})
        for _, row in responsive.iterrows():
            st = spikes_grp[row["hdf5_key"]][...]
            counts, _ = np.histogram(st, bins=edges) if st.size else (np.zeros_like(time_vec), _)
            raw_rate_df[row["hdf5_key"]] = counts.astype(float) / dt  # Hz

    # z-score across time for each neuron
    rate_cols = [c for c in raw_rate_df.columns if c != "time_s"]
    scaler    = StandardScaler()
    z_rates   = scaler.fit_transform(raw_rate_df[rate_cols].values)

    z_df = pd.DataFrame(z_rates, columns=rate_cols)
    z_df.insert(0, "time_s", raw_rate_df["time_s"].values)

    # normalise speed & build 0/1 foot-shock vector 
    speed_filled = np.nan_to_num(speed, nan=0.0).reshape(-1, 1)
    speed_scaled = MinMaxScaler().fit_transform(speed_filled).flatten()

    footshock_vec = np.zeros_like(time_vec, dtype=int)
    tol = dt / 2
    for ts in footshock_times:
        if time_vec[0] <= ts <= time_vec[-1]:
            idx = np.argmin(np.abs(time_vec - ts))
            if abs(time_vec[idx] - ts) <= tol:
                footshock_vec[idx] = 1
    #save files
    raw_csv = out_dir / "responsive_rates_raw.csv"
    z_csv   = out_dir / "responsive_rates_z.csv"
    meta_csv= out_dir / "responsive_metadata.csv"

    raw_rate_df.to_csv(raw_csv, index=False)
    z_df.to_csv(z_csv,           index=False)
    responsive.to_csv(meta_csv,  index=False)

    if verbose:
        print(f"✓ wrote {raw_csv.name}, {z_csv.name}, {meta_csv.name}")
    # return arrays
    return dict(
        time        = time_vec,                 # (T,)
        speed_scaled= speed_scaled,             # (T,)
        footshock   = footshock_vec,            # (T,)
        rates_raw   = raw_rate_df[rate_cols].values,  # (T, N)
        rates_z     = z_rates,                        # (T, N)
        metadata    = responsive.reset_index(drop=True)
    )
