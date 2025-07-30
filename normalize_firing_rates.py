from sklearn.preprocessing import StandardScaler  

__all__ = ["normalize_firing_rates"]

def normalize_firing_rates(fr, *, time_first=True, return_scaler=False):
    """
    Z-score each neuron's firing-rate vector independently.
    ...
    """
    X = fr if time_first else fr.T            # ensure (samples, features)

    scaler = StandardScaler()                 # now defined
    Xz     = scaler.fit_transform(X)

    fr_z = Xz if time_first else Xz.T
    if return_scaler:
        return fr_z, scaler
    return fr_z
