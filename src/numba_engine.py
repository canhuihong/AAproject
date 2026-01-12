import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def run_parallel_ols(Y, X, min_obs=50):
    """
    Parallel OLS Helper using Numba.
    
    Parameters:
    -----------
    Y : np.array (T, N)
        Excess returns for N stocks over T days.
        Columns corresponding to tickers.
    X : np.array (T, K)
        Factor returns (Design Matrix). Constant column MUST be included by caller.
    min_obs : int
        Minimum non-NaN observations required to run regression.
        
    Returns:
    --------
    results : np.array (N, K+2)
        Rows: Tickers
        Cols: [Alpha (Annualized), Volatility (Annualized), Beta_0, Beta_1, ..., Beta_K]
               Note: Alpha is essentially Beta_const * 252 (if const is first col)
    """
    T, N = Y.shape
    _, K = X.shape
    
    # Pre-allocate output matrix
    # Format: [Alpha, Volatility, Betas...]
    # K includes constant, so we have K betas.
    # Output cols: 2 + K
    out = np.full((N, K + 2), np.nan)
    
    for i in prange(N):
        # Extract single stock return stream
        y = Y[:, i]
        
        # 1. Create mask for valid data (Y and X must both be valid)
        # Assuming X is usually full, but let's be safe.
        # Numba doesn't list comprehension well here, use explicit loop or array ops
        mask = np.isfinite(y) 
        
        # Check sufficient history
        valid_count = 0
        for t in range(T):
            if mask[t]:
                valid_count += 1
                
        if valid_count < min_obs:
            continue
            
        # 2. Slice data
        # We need to construct contiguous arrays for lstsq
        y_clean = np.empty(valid_count, dtype=np.float64)
        X_clean = np.empty((valid_count, K), dtype=np.float64)
        
        curr = 0
        for t in range(T):
            if mask[t]:
                y_clean[curr] = y[t]
                # Manual copy row
                for k in range(K):
                    X_clean[curr, k] = X[t, k]
                curr += 1
                
        # 3. Solve OLS: (X'X)^-1 X'Y
        # Manual implementation to be thread-safe in parallel mode
        # Numba's lstsq can have issues with some BLAS backends in parallel sections
        
        # Xt = X_clean.T
        # XtX = Xt @ X_clean
        # Xty = Xt @ y_clean
        # beta = np.linalg.solve(XtX, Xty)
        
        # Manual matrix multiplication for maximum safety/compatibility
        Xt = X_clean.T
        XtX = np.dot(Xt, X_clean)
        Xty = np.dot(Xt, y_clean)
        
        # [Fix] Add Ridge Regularization (L2) to ensure invertibility
        # This replaces the need for try...except (which breaks Numba parallel)
        for k in range(K):
            XtX[k, k] += 1e-9
            
        # Solve (safe now due to Ridge)
        beta = np.linalg.solve(XtX, Xty)

        # 4. Calculate Stats
        # Beta is [Const, Beta1, Beta2...]

        # Alpha (Annualized) -> Assuming First Column of X is Constant
        alpha_ann = beta[0] * 252
        
        # Volatility (Annualized Standard Deviation of Raw Returns)
        # The user's original code used `np.std(y) * np.sqrt(252)` (Total Risk), not residual risk.
        # We assume sample std dev (ddof=1) or pop? np.std is pop by default with numpy.
        # Let's compute manually to match logic. 
        # Mean calculation
        sum_y = 0.0
        for val in y_clean:
            sum_y += val
        mean_y = sum_y / valid_count
        
        sum_sq_diff = 0.0
        for val in y_clean:
            sum_sq_diff += (val - mean_y)**2
            
        # Match numpy std (population)
        std_dev = np.sqrt(sum_sq_diff / valid_count)
        vol_ann = std_dev * np.sqrt(252)
        
        # 5. Store Results
        out[i, 0] = alpha_ann
        out[i, 1] = vol_ann
        for k in range(K):
            out[i, 2 + k] = beta[k]
            
    return out
