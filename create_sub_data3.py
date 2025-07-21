import numpy as np, pandas as pd, h5py, os
from pathlib import Path
from metadata import match_units_to_hdf5       

__all__ = ["export_area_rate_filtered"]

def export_area_rate_filtered(base_dir,
                              rat_id,
                              units_csv,
                              *,
                              h5_prefix    = "NpxFiringRate",
                              spike_group  = "spike_times",
                              min_rate_hz  = 0.5,      # keep ≥ 0.5 Hz
                              out_dir      = None,
                              verbose      = True):
    """
    For each macro-area (dorsal / ventral / thalamus) export two CSVs
    containing only neurons whose mean firing rate ≥ `min_rate_hz`.

    Returns
    -------
    dict  { area : [wide_path, long_path], ... }
    """
    rat_tag = f"Rat{int(rat_id)}"
    mapping = match_units_to_hdf5(units_csv, base_dir, rat_id,
                                  h5_prefix=h5_prefix, verbose=False)


    rat_dir = Path(base_dir) / rat_tag
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_path, "r") as f:
        time_vec = f["time"][...]
        speed    = f["speed"][...] if "speed" in f else None
        rates_all= f["firing_rates"][...]        # (n_units × T)
        spikes   = f[spike_group]

    rec_len_s = time_vec[-1] - time_vec[0]       # ≈ session duration (s)
    spike_counts = np.array([spikes[k].size for k in spikes.keys()])
    mean_rate_hz = spike_counts / rec_len_s

    # append mean rate to mapping
    mapping["mean_rate_hz"] = mean_rate_hz[mapping.firing_rate_index.values]

    # keep only ≥ threshold
    mapping = mapping[mapping.mean_rate_hz >= min_rate_hz]
    if verbose:
        print(f"{rat_tag}: units ≥ {min_rate_hz} Hz → {len(mapping)}")

    # output folder
    if out_dir is None:
        out_dir = rat_dir / f"{rat_tag}_area_{min_rate_hz:.2f}Hz"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    exported = {}
    for area in sorted(mapping.macro.unique()):
        sub = mapping[mapping.macro == area]
        if sub.empty:
            continue

        if verbose:
            print(f"[{area}] {len(sub)} neurons ≥ {min_rate_hz} Hz")

        # wide format (time × neurons) 
        wide = pd.DataFrame(rates_all[sub.firing_rate_index].T,
                            columns=sub.cluster.astype(str))
        wide.insert(0, "time_s", time_vec)
        if speed is not None:
            wide["speed"] = speed
        wide_path = out_dir / f"{area}_wide.csv"
        wide.to_csv(wide_path, index=False)

        # long (tidy) format
        recs = []
        for _, row in sub.iterrows():
            rates = rates_all[row.firing_rate_index]
            recs.extend({
                "time_s"     : t,
                "cluster"    : row.cluster,
                "area"       : row.area,
                "macro_area" : area,
                "mean_rate_hz": row.mean_rate_hz,
                "rate"       : r,
                "speed"      : s if speed is not None else np.nan
            } for t, r, s in zip(time_vec, rates, speed if speed is not None else np.repeat(np.nan, len(rates))))
        long_path = out_dir / f"{area}_long.csv"
        pd.DataFrame.from_records(recs).to_csv(long_path, index=False)

        exported[area] = [str(wide_path), str(long_path)]
        if verbose:
            print(f"[{area}] files → {wide_path.name}, {long_path.name}")

    if verbose:
        print(f"✓ Saved in {out_dir}/")
    return exported
