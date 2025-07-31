import pandas as pd
import numpy as np

def csv_wide_to_rates(path):
    df = pd.read_csv(path, index_col=0)     # first col = time or index
    rates = df.to_numpy(dtype=np.float32).T # now [N, T]
    return rates
