import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from numpy.typing import NDArray

class EventDetector:
    """
    Class for detecting runoff events in a discharge time series.
    """
    def __init__(
        self,
        runoff_threshold: float = 0.01,
        event_threshold: float = 0.2,
        max_nans: float = 0.2,
        min_peakness: float = 0.6,
        filter_type: Union[str, int] = 1,
        fparam: float = 0.995,
        bfi: float = 0.8,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the detector with filtering and event criteria.

        Args:
            runoff_threshold: below this (in filtered runoff) is zeroed out
            event_threshold: minimum relative peak height
            max_nans: max fraction of missing data allowed in an event
            min_peakness: (peak - start) / peak must exceed this
            filter_type: 1=Eckhardt, 2=Nathan, 3=Chapman
            fparam: filter parameter (alpha or equivalent)
            bfi: baseflow index for Eckhardt filter
            verbose: logging
        """
        self.runoff_threshold = runoff_threshold
        self.event_threshold = event_threshold
        self.max_nans = max_nans
        self.min_peakness = min_peakness
        self.filter_type = filter_type
        self.fparam = fparam
        self.bfi = bfi
        self.verbose = verbose

    def detect_events(
        self,
        discharge: pd.DataFrame,
        only_events: bool = True
    ) -> Union[List[pd.DataFrame], Tuple[List[pd.DataFrame], pd.DataFrame, np.ndarray]]:
        """
        Main entry: apply filter, find and screen event windows.

        Args:
            discharge: DataFrame with datetime index and one column of flow.
            only_events: if True, return just event windows list.

        Returns:
            List of event windows or (events, full_filter_df, binary_mask_array).
        """
        # Copy and interpolate
        df = discharge.copy()
        col = df.columns[0]
        df['mask'] = df[col].isna()
        df[col] = df[col].interpolate(limit_direction='both')
        if self.verbose:
            print(f"-[Detect]: Data read for dataframe col: {col}")

        # Apply chosen digital filter
        filtered = self._apply_filter(df[col])

        # Zero small runoff
        if self.verbose:
            print("- [Detect]: Thresholding Runoff...")
        filtered.loc[filtered['Runoff'] < self.runoff_threshold, 'Runoff'] = 0

        # Identify contiguous positive-runoff blocks
        if self.verbose:
            print("- [Detect]: Runoff Blocks: Binarizing...")
        idx, bin_mask = self._binarize_ts(filtered["Runoff"], df)

        # Screen windows
        if self.verbose:
            print("- [Detect]: Selecting events...")
        events = self._separate_events(df, idx)

        if only_events:
            if self.verbose:
                print("- [OUTPUT]: Events")
            return events
        
        if self.verbose:
            print("- [OUTPUT]: Events and streamflow dataframe")
        return events, filtered, bin_mask

    def _apply_filter(self, series: pd.Series) -> pd.DataFrame:
        """
        Route to the specific filter implementation.
        """
        q = series.values
        if self.filter_type in ('Eckhardt', 1):
            if self.verbose:
                print("--[Filter] Applying Eckhardt (2005)...")
            runoff, baseflow = self._eckhardt2005(q)
        elif self.filter_type in ('Nathan', 2):
            if self.verbose:
                print("--[Filter] Applying Nathan & McMahon (1990)...")
            runoff, baseflow = self._nathan1990(q)
        elif self.filter_type in ('Chapman', 3):
            if self.verbose:
                print("--[Filter] Applying Chapman & Maxwell (1996)...")
            runoff, baseflow = self._chapmanmaxwell1996(q)
        else:
            raise ValueError(f'Unknown filter type: {self.filter_type}')
            
        if self.verbose:
            print("--[Filter]: Process complete!")
        return pd.DataFrame({'Runoff': runoff, 'Baseflow': baseflow}, index=series.index)
    
    def _binarize_ts(self, 
                     runoff_df: Union[pd.Series, pd.DataFrame], 
                     q_df: Union[pd.Series, pd.DataFrame]) -> pd.Series:

        
        z = runoff_df.values
        k = (z > 0).astype(np.int8)    # binary mask
        dk = np.diff(k)                # sign change in binary mask
        p = np.where(dk == 1)[0]       # start of the events
        return q_df.index[p], k

    def _separate_events(self, q, idx):

        events = []
        for i in range(len(idx) - 1):
            if q.loc[idx[i]:idx[i + 1], q.columns[0]].max() > self.event_threshold:
                _a = q[idx[i]:idx[i + 1]]
                if _a.loc[_a['mask'] == True].shape[0] / _a.shape[0] < self.max_nans:
                    _a = _a[q.columns[0]]
                    peakness = (_a.max() - _a.values[0]) / _a.max()
                    if peakness > self.min_peakness:
                        events.append(q[idx[i]:idx[i + 1]])

        return events


    def _eckhardt2005(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eckhardt recursive baseflow filter (Eckhardt 2005).

        Args:
            q: array of discharge values
        Returns:
            runoff and baseflow arrays
        """
        
        baseflow = np.zeros_like(q)
        alpha = self.fparam
        bfi = self.bfi

        if not (0.0 < alpha < 1.0):
          raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if not (0.0 < bfi < 1.0):
          raise ValueError(f"BFI must be in (0,1), got {bfi}")
        
        denom = 1.0 - alpha * bfi
        if abs(denom) < 1e-12:
            raise ValueError(f"1 - alpha*BFI too small: alpha={alpha}, BFI={bfi}")

        # First time step
        baseflow[0] = q[0]
        for i in range(1, len(q)):
            b = ((1.0 - bfi) * alpha * baseflow[i-1] + (1.0 - alpha)* bfi * q[i]) / denom
            b = b if b > 0.0 else 0.0
            baseflow[i] = q[i] if b > q[i] else b

        runoff = q - baseflow

        return runoff, baseflow

    def _nathan1990(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        One parameter digital filter of Nathan and McMahon (1990)
        Args:
            q: array of discharge values
        Returns:

        """
        runoff = np.zeros(q.size)
        a = self.fparam

        for c, (q1, q2) in enumerate(zip(q[:-1], q[1:]), start=1):
            runoff[c] = a * runoff[c - 1] + ((1 + a) / 2.) * (q2 - q1)
            if runoff[c] < 0:
                runoff[c] = 0
            elif runoff[c] > q2:
                runoff[c] = q2

        baseflow = q - runoff
        return runoff, baseflow

    def _chapmanmaxwell1996(self, q: NDArray[np.floating]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Digital filter proposed by chapman and maxwell (1996)
        """
        q = np.asarray(q, dtype=float)
        a: float = self.fparam
        b: NDArray[np.floating] = np.zeros_like(q)
        coef_prev: float = a / (2. - a)
        coef_curr: float = (1. - a) / (2. - a)

        for idx in range(1, q.shape[0]):
            b[idx] = coef_prev * b[idx - 1] + coef_curr * float(q[idx])
        r = q - b
        return r, b
