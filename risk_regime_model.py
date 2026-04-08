import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import Callable, Optional

# --- 1. Data Structures and Parameters ---

class Regime(IntEnum):
    LOW = -1
    NEUTRAL = 0
    HIGH = 1
    UNKNOWN = 999

@dataclass
class EwmaVolRegimeParams:
    """Parameters inferred from the video snippets (approx 02:53)"""
    lam: float = 0.94              # EWMA decay factor
    init_size: int = 20            # Days to initialize variance
    window: int = 252              # Rolling window for quantiles (e.g., 1 trading year)
    q_low: float = 0.20            # 20th percentile threshold
    q_high: float = 0.80           # 80th percentile threshold
    quantile_method: str = "linear"

# --- 2. Helper Functions ---

def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Computes daily close-to-close log returns."""
    rt = np.full_like(prices, np.nan, dtype=float)
    # rt[t] corresponds to r_t with t as 0-based index
    prev = prices[:-1]
    curr = prices[1:]
    
    # Only compute where both prices are valid and > 0
    valid = np.isfinite(prev) & np.isfinite(curr) & (prev > 0.0) & (curr > 0.0)
    rt[1:][valid] = np.log(curr[valid] / prev[valid])
    return rt

# --- 3. Core Logic Extracted from Video ---

def compute_ewma_vol(prices: np.ndarray, params: EwmaVolRegimeParams) -> np.ndarray:
    """
    Extracted from 03:07.
    Computes Exponentially Weighted Moving Average Volatility.
    """
    rt = compute_log_returns(prices)
    T = len(prices)
    
    sigma = np.full(T, np.nan, dtype=float)
    sigma2 = np.full(T, np.nan, dtype=float)
    
    k = params.init_size
    lam = params.lam
    
    if T <= k:
        return sigma
        
    # Initialization at index k
    init_slice = rt[1 : k + 1]
    if len(init_slice) > 0 and np.any(np.isfinite(init_slice)):
        s2 = np.nanvar(init_slice, ddof=1)
        sigma2[k] = s2
        sigma[k] = np.sqrt(s2)
    else:
        return sigma

    # Forward recursion from t=k+1
    prev_s2 = sigma2[k]
    
    for t in range(k + 1, T):
        r = rt[t]
        
        if not np.isfinite(prev_s2):
            # If we lost initialization, keep NaN throughout
            sigma2[t] = np.nan
            sigma[t] = np.nan
            continue
            
        if np.isfinite(r):
            s2 = lam * prev_s2 + (1.0 - lam) * (r * r)
        else:
            # Missing data policy: skip update
            s2 = prev_s2
            
        sigma2[t] = s2
        sigma[t] = np.sqrt(s2)
        prev_s2 = s2
        
    return sigma

def compute_regimes(volatility: np.ndarray, params: EwmaVolRegimeParams) -> np.ndarray:
    """
    Extracted/Reconstructed from 04:02.
    Calculates rolling quantiles to classify volatility into relative regimes.
    """
    T = len(volatility)
    regimes = np.full(T, Regime.UNKNOWN, dtype=int)
    
    window = params.window
    
    for t in range(T):
        v = volatility[t]
        if not np.isfinite(v):
            continue
            
        # Define rolling window
        start_idx = max(0, t - window + 1)
        buf = volatility[start_idx : t + 1]
        
        # Filter NaNs from buffer
        buf = buf[np.isfinite(buf)]
        
        if len(buf) < window * 0.5: # Require at least half a window to start
            continue
            
        L = np.quantile(buf, params.q_low, method=params.quantile_method)
        H = np.quantile(buf, params.q_high, method=params.quantile_method)
        
        # Inclusive inequality rules as shown in the video
        if v <= L:
            regimes[t] = int(Regime.LOW)
        elif v >= H:
            regimes[t] = int(Regime.HIGH)
        else:
            regimes[t] = int(Regime.NEUTRAL)
            
    return regimes

def default_mapping(r: int) -> float:
    """Default risk multipliers based on regime."""
    if r == Regime.LOW:
        return 1.0     # Trade normal
    if r == Regime.NEUTRAL:
        return 0.8     # Trade slightly smaller
    if r == Regime.HIGH:
        return 0.5     # Trade small
    return 0.0

def regimes_to_actions(
    regimes: np.ndarray, 
    mapping: Optional[Callable[[int], float]] = None,
    offset: int = 1
) -> np.ndarray:
    """
    Extracted from 04:32.
    Maps regimes to actions WITH a strict 1-day lag to prevent look-ahead bias.
    """
    if mapping is None:
        mapping = default_mapping
        
    actions = np.full_like(regimes, np.nan, dtype=float)
    T = len(regimes)
    
    # Crucial step: t represents the day the action is taken.
    # We look at the regime from t-offset (usually yesterday).
    for t in range(offset, T):
        r = regimes[t - offset]
        actions[t] = float(mapping(r))
        
    return actions

def compute_regime_and_action(prices: np.ndarray) -> tuple:
    """
    Master function shown briefly at 04:42 orchestrating the pipeline.
    """
    params = EwmaVolRegimeParams()
    
    # 1. Get daily EWMA Volatility
    vol = compute_ewma_vol(prices, params)
    
    # 2. Translate absolute vol to relative regimes (Low, Neutral, High)
    regimes = compute_regimes(vol, params)
    
    # 3. Map to actions with a strict 1-day lag
    actions = regimes_to_actions(regimes)
    
    return vol, regimes, actions

# --- Example Usage ---
if __name__ == "__main__":
    # Generate mock price data (random walk)
    np.random.seed(42)
    mock_returns = np.random.normal(0.001, 0.02, 1000)
    mock_prices = np.exp(np.cumsum(mock_returns)) * 100
    
    volatility, regimes, actions = compute_regime_and_action(mock_prices)
    
    print("Pipeline executed successfully.")
    print(f"Sample Action for day 500 (Multiplier): {actions[500]}")