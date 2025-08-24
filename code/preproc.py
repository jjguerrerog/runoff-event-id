import pandas as pd

def procs(q, k):
    """
    q: dataframe with the discharge data
    k: column name of the discharge data (USGS gauge ID)
    returns: normalized discharge data for a given gauge
    """
    _q = q[k].copy()  # Copy the column
    # negatives turn to nan and drop them
    _q = _q.resample('1h').max()  # Resample to 1-hourly, taking max
    annual_max_mean = _q.resample('YE').max().mean()  # Mean of annual maxima
    if pd.isna(annual_max_mean) or annual_max_mean == 0:
        # Handle edge cases: return NaN series or raise warning if mean is 0 or NaN
        return pd.Series(index=_q.index, dtype=float) * float('nan')
    _q = _q / annual_max_mean  # Normalize
    return _q

def process_dataframe(q):
    """
    q: dataframe with the discharge data (several gauges)
    returns: normalized discharge data for all gauges
    """
    # Ensure index is datetime
    if not isinstance(q.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    # Create an empty DataFrame to store results
    q_df = pd.DataFrame(index=q.resample('1h').max().index)

    # Iterate over columns
    for k in q.columns:
        q_df[k] = procs(q, k)

    return q_df