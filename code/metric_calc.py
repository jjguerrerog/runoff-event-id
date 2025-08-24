import numpy as np
import pandas as pd
from tqdm import tqdm

def _std_metrics(series: pd.Series, window: int):
    """
    Compute global std and rolling std.
    """
    global_stat = series.std(ddof=1)
    rolling_stat = series.rolling(window=window).std(ddof=1)
    return global_stat, rolling_stat

def _iqr_metrics(series: pd.Series, window: int, pctiles=(0.25, 0.75)):
    """
    Compute global IQR and rolling IQR via two quantiles.
    """
    q_low, q_high = pctiles
    global_stat = series.quantile(q_high) - series.quantile(q_low)

    # vectorized rolling IQR
    rolling_q_high = series.rolling(window=window).quantile(q_high)
    rolling_q_low  = series.rolling(window=window).quantile(q_low)
    rolling_stat = rolling_q_high - rolling_q_low
    return global_stat, rolling_stat

STAT_HELPERS = {
    'std': _std_metrics,
    'iqr': _iqr_metrics,
}

def norm_roll_metric(
    df: pd.DataFrame,
    column: str = 'Q',
    stat: str = 'std',
    agg: str = 'mean',
    window_hours: int = 168,
    pctiles: tuple[float, float] = (0.25, 0.75), # used for IQR
) -> float:
    """
    Compute a normalized rolling metric on a DataFrame column.

    metric = (aggregated rolling statistic) / (global statistic)

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `column`.
    column : str
        Name of the numeric column to analyze.
    stat : {'std', 'iqr'}
        Which statistic to compute.
    agg : {'mean', 'median'}
        How to aggregate the rolling-statistic series.
    window_hours : int
        Number of rows (e.g. hours) in the rolling window.
    pctiles : tuple of float
        Only used for iqr: (lower_quantile, upper_quantile).

    Returns
    -------
    float
        The ratio. Returns 0 if the global statistic is zero.
    """
    stat_key = stat.lower()
    if stat_key not in STAT_HELPERS:
        raise ValueError(f"Unknown stat `{stat}`; choose 'std' or 'iqr'.")

    # compute global & rolling series
    helper = STAT_HELPERS[stat_key]
    if stat_key == 'iqr':
        global_stat, rolling_stat = helper(df[column], window_hours, pctiles)
    else:
        global_stat, rolling_stat = helper(df[column], window_hours)

    # aggregate
    if agg == 'mean':
        agg_val = rolling_stat.mean()
    elif agg == 'median':
        agg_val = rolling_stat.median()
    else:
        raise ValueError(f"Unknown agg `{agg}`; choose 'mean' or 'median'.")

    # handle zero global stat
    if global_stat == 0 or pd.isna(global_stat):
        return 0.0

    return agg_val / global_stat



def _compute_for_window(qs: pd.DataFrame, window_hours: list, win_label: list, 
                        stats: list = ['std'], agg: str='median'):
    """
    Compute metrics for a single window size.

    Returns a DataFrame of shape (len(qs), len(stats)), indexed by gauge,
    with columns named after the stats.
    
    stats is either a list or a tuple (even if it is just one statistic)
    """
    mets = {}
    desc = f"Processing Gauges for window = {win_label} weeks"
    with tqdm(total=len(qs), desc=desc, unit="gauge") as pbar:
        for gauge_id, q in qs.items():
            pbar.set_description(f"Gauge {gauge_id}")
            pbar.set_postfix({"Status": f"window={win_label}-wks"})

            # compute each stat for this gauge
            row = {
                    stat: norm_roll_metric(
                        q.to_frame(gauge_id), column=gauge_id, stat=stat, 
                        agg=agg, window_hours=window_hours
                    )
                    for stat in stats
                }
            mets[gauge_id] = pd.Series(row, name=gauge_id)

            pbar.update(1)

    # assemble into a DataFrame: gauges × stats
    df = pd.concat(mets.values(), axis=1).T
    return df

def compute_metrics(
    qs: pd.DataFrame,
    windows: list[int],
    week_labels: list[float],
    stats: tuple[str,str]=('std','iqr'),
    agg: str='median',
) -> dict[str, pd.DataFrame]:
    """
    Compute norm_roll_metric for multiple window sizes.

    Parameters
    ----------
    qs : dict
        {gauge_id: DataFrame_of_streamflow}
    windows : list of int
        rolling-window sizes in hours.
    week_labels : list of float
        labels (in weeks) matching `windows`.
    stats : tuple of str
        which stats to compute (default ('std','cv')).
    agg : {'mean','median'}
        how to aggregate rolling series before normalizing.

    Returns
    -------
    dict
        keys 'S0.25','S0.5',… mapping to DataFrames (gauges × stats).
    """
    if len(windows) != len(week_labels):
        raise ValueError("`windows` and `week_labels` must be same length")

    metrics = {}
    for win_label, window in zip(week_labels, windows):
        key = f"S{win_label}"
        df = _compute_for_window(qs, window, win_label, stats=stats,
                                 agg=agg)
        metrics[key] = df

    return metrics
