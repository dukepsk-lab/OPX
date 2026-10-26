"""Statistical tests and significance metrics."""

import itertools
import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew


def wilson_lcb(wins: int, n: int, conf: float = 0.95) -> float:
    """One-sided Wilson score lower confidence bound for a binomial proportion.
    
    Returns nan if n == 0.
    """
    if n <= 0:
        return np.nan

    z = float(norm.ppf(conf))
    ph = float(wins) / float(n)
    z2 = z ** 2

    numerator = ph + z2 / (2.0 * n) - z * np.sqrt(ph * (1.0 - ph) / n + z2 / (4.0 * (n ** 2)))
    denominator = 1.0 + z2 / n

    return float(numerator / denominator)


def deflated_sharpe(returns: np.ndarray, n_trials: int) -> float:
    """Deflated Sharpe Ratio (Bailey and López de Prado).
    
    Computes the probability that the estimated Sharpe Ratio exceeds the expected
    maximum Sharpe Ratio under the null of no skill, accounting for trial count,
    skewness, and kurtosis.
    """
    ret = np.asarray(returns, dtype=float)
    T = len(ret)
    if T < 2:
        return 0.0

    std = float(np.std(ret, ddof=1))
    if std <= 1e-12:
        return 0.0

    mean = float(np.mean(ret))
    SR = mean / std

    N = max(1, int(n_trials))
    gamma_e = 0.5772156649

    if N == 1:
        SR0 = 0.0
    else:
        term1 = (1.0 - gamma_e) * norm.ppf(1.0 - 1.0 / N)
        term2 = gamma_e * norm.ppf(1.0 - 1.0 / (N * np.e))
        SR0 = np.sqrt(1.0 / (T - 1.0)) * (term1 + term2)

    gamma3 = float(skew(ret, bias=False))
    gamma4 = float(kurtosis(ret, fisher=False, bias=False))

    denom_var = 1.0 - gamma3 * SR + ((gamma4 - 1.0) / 4.0) * (SR ** 2)
    if denom_var <= 0.0:
        denom_var = 1e-12

    z = (SR - SR0) * np.sqrt(T - 1.0) / np.sqrt(denom_var)
    dsr = float(norm.cdf(z))
    return float(np.clip(dsr, 0.0, 1.0))


def permutation_pvalue(
    y: np.ndarray,
    take: np.ndarray,
    r: float,
    n_perm: int = 2000,
    seed: int = 0,
) -> float:
    """Permutation test p-value of EV on selected trades.
    
    Shuffles y n_perm times and tests how often permuted EV >= observed EV.
    Returns nan if take is empty.
    """
    take_arr = np.asarray(take, dtype=bool)
    if not np.any(take_arr) or len(y) == 0:
        return np.nan

    y_arr = np.asarray(y, dtype=float)
    obs_pnl = np.where(y_arr[take_arr] == 1.0, r, -1.0)
    obs_ev = float(np.mean(obs_pnl))

    rng = np.random.default_rng(seed)
    unique_vals = set(np.unique(y_arr))
    if unique_vals.issubset({0.0, 1.0}):
        K = int(np.sum(take_arr))
        M = int(np.sum(y_arr == 1.0))
        N = len(y_arr)
        draws = rng.hypergeometric(ngood=M, nbad=N - M, nsample=K, size=n_perm)
        perm_evs = (draws * r - (K - draws)) / K
        count = int(np.sum(perm_evs >= obs_ev))
        return float((1.0 + count) / (1.0 + n_perm))

    y_perm = y_arr.copy()
    count = 0

    for _ in range(n_perm):
        rng.shuffle(y_perm)
        perm_ev = np.mean(np.where(y_perm[take_arr] == 1.0, r, -1.0))
        if perm_ev >= obs_ev:
            count += 1

    return float((1.0 + count) / (1.0 + n_perm))


def pbo(perf: np.ndarray, n_blocks: int = 16) -> float:
    """Combinatorially Symmetric Cross-Validation (CSCV) Probability of Backtest Overfitting.
    
    perf is shape (T, n_configs) of per-period returns.
    """
    perf_arr = np.asarray(perf, dtype=float)
    if perf_arr.ndim != 2:
        raise ValueError(f"perf must be 2D (T, n_configs), got shape {perf_arr.shape}")

    T, n_configs = perf_arr.shape
    if n_configs <= 1:
        return np.nan
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even, got {n_blocks}")
    if T < n_blocks:
        raise ValueError(f"T ({T}) must be >= n_blocks ({n_blocks})")

    blocks = np.array_split(np.arange(T), n_blocks)
    block_sums = np.array([perf_arr[b].sum(axis=0) for b in blocks])
    block_lens = np.array([len(b) for b in blocks])

    half = n_blocks // 2
    combos = np.array(list(itertools.combinations(range(n_blocks), half)))

    is_sums = block_sums[combos].sum(axis=1)
    is_lens = block_lens[combos].sum(axis=1)[:, None]
    is_means = is_sums / is_lens

    best_configs = np.argmax(is_means, axis=1)

    total_sum = block_sums.sum(axis=0)
    total_len = T

    oos_sums = total_sum - is_sums
    oos_lens = total_len - is_lens
    oos_means = oos_sums / oos_lens

    ranks = rankdata(oos_means, axis=1)
    best_ranks = ranks[np.arange(len(combos)), best_configs]

    w = best_ranks / (n_configs + 1.0)
    # lambda = ln(w / (1 - w)) <= 0 iff w <= 0.5
    return float(np.mean(w <= 0.5))
