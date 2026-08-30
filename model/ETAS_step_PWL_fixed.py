import numpy as np
import pandas as pd
from tqdm import tqdm

# np.random.seed(1)
LOG_FILE = "./results_intgerate/intensity_anomalies.log"

def k(m, params, Mcut):
    if(isinstance(m, (list, tuple, np.ndarray,pd.Series))):
        x = np.where(m >= Mcut, params['k0'] * np.exp(params['a'] * (m - params['M0'])), 0)
    else:
        if(m >= Mcut):
            x = params['k0'] * np.exp(params['a'] * (m - params['M0']))
        else:
            x = 0
    return x

def f(x, params):
    return (params['omega'] - 1) * params['c'] ** (params['omega'] - 1) * 1 / ((x + params['c']) ** params['omega'])

def H(t, params):
    if(isinstance(t, (list, tuple, np.ndarray, pd.Series))):
            x = np.where(t >= 0, 1 - params['c'] ** (params['omega'] - 1) / (t + params['c']) ** (params['omega'] - 1), 0)
    else:
        if(t >= 0):
            x = 1 - params['c'] ** (params['omega'] - 1) / (t + params['c']) ** (params['omega'] - 1)
        else:
            x = 0
    return x

def intensity_vectorized(t, T_hist, M_hist, params, Mcut):
    T_hist = np.array(T_hist)
    M_hist = np.array(M_hist)
    dt = t - T_hist
    valid = (dt + params['c']) > 0  # 确保不会除以0或负数

    if not np.any(valid):
        return params['mu']  # 所有值都无效时只返回背景率

    km = k(M_hist[valid], params, Mcut)
    fm = f(dt[valid], params)
    cumulative = np.sum(km * fm)
    return params['mu'] + cumulative

# ============================================================
#  ETAS-PWL: 时间部分
# ============================================================

def etas_time_input_pwl(Tdat, Mdat, Mcut, M0pred, params_3, time_step, Mmax):
    # ---- 读参数 / 常量 ----
    M0   = float(params_3['M0'])
    mu_3   = float(params_3['mu'])
    beta_3 = float(params_3['beta'])
    
    Mmax = float(Mmax)
    Delta = Mmax - Mcut
    delta = M0pred - Mcut
    gamma = 1.5 * np.log(10.0)
    alpha = gamma - beta_3
    Z_norm = 1.0 - np.exp(-beta_3 * Delta)

    w_m_full = np.exp(gamma * (Mdat - M0pred))
    C_wd = (
            beta_3 * np.exp(gamma * (Mcut - M0pred)) * (np.exp(alpha * Delta) - np.exp(alpha * delta))
        ) / (
            alpha * (np.exp(-beta_3 * delta) - np.exp(-beta_3 * Delta))
        )

    p_dw = (np.exp(-beta_3 * delta) - np.exp(-beta_3 * Delta)) / Z_norm

    if abs(Mcut - M0pred) < 1e-12:
        p_3_cut = 1.0
    else:
        p_3_cut = (np.exp(-beta_3 * delta) - np.exp(-beta_3 * Delta)) / Z_norm


    # ---- 基本检查 ----
    assert Tdat.ndim == 1 and Mdat.ndim == 1 and len(Tdat) == len(Mdat), "Tdat/Mdat must be 1D & same length."
    assert np.all(np.diff(Tdat) > 0), "Tdat must be strictly increasing."

    n = len(Tdat)
    # ---- 全量预分配（float64）----
    lam_vals_full      = np.zeros(n, dtype=np.float64)
    integral_vals_full = np.zeros(n, dtype=np.float64)
    loglikelihood_full = np.zeros(n, dtype=np.float64)
    integral_step_full = np.zeros(n, dtype=np.float64)

    # ---- 主循环（累计口径 + 单步口径）----
    for i in tqdm(range(1, n), desc='ETAS time LL (vec)', ncols=100):
        lam_i = intensity_vectorized(Tdat[i], Tdat[:i], Mdat[:i], params_3, M0)
        w_i = w_m_full[i]

        lam_vals_full[i]      = lam_i / p_3_cut
        loglikelihood_full[i] = w_i * np.log(lam_i)   

        if np.any(Mdat[:i] >= M0pred):
            j = np.where(Mdat[:i] >= M0pred)[0][-1]
        else:
            j = 0  

        # ===== 累计口径的积分（背景 + 触发）=====
        dt_mu = Tdat[i] - Tdat[j]
        bg_cum_3 = (mu_3 / p_3_cut) * dt_mu
        bg_cum = mu_3 * dt_mu
        loglikelihood_full[i] -= C_wd * bg_cum
        integral_vals_full[i] += bg_cum_3

        k_i = k(Mdat[:i], params_3, M0)                       # shape: (i,)
        H_i = H(Tdat[i] - Tdat[:i], params_3)                 # shape: (i,)
        sum_kH_i = np.sum(k_i * H_i)

        if j == 0:
            trig_cum = sum_kH_i
        else:
            k_j = k(Mdat[:j], params_3, M0)                   # shape: (j,)
            H_j = H(Tdat[j] - Tdat[:j], params_3)             # shape: (j,)
            trig_cum = sum_kH_i - np.sum(k_j * H_j)

        loglikelihood_full[i] -= C_wd * trig_cum
        integral_vals_full[i] += trig_cum / p_3_cut

        # ===== 单步积分（区间 (t_{i-1}, t_i]），仅用于核验/可视化 =====
        dt = Tdat[i] - Tdat[i-1]
        step_bg = (mu_3 / p_3_cut) * dt
        H_im1 = H(Tdat[i-1] - Tdat[:i], params_3)             # shape: (i,)
        step_trig = np.sum(k_i * (H_i - H_im1)) / p_3_cut

        integral_step_full[i] = step_bg + step_trig

    # ---- 现在再 cut ----
    cut = int(time_step) + 1
    lam_vals      = lam_vals_full[cut:]
    integral_vals = integral_vals_full[cut:]
    integral_step = integral_step_full[cut:]
    mask          = (Mdat >= M0pred)

    time_ll_pred = loglikelihood_full[cut:][mask[cut:]]

    # ---- 返回（与之前一致 + integral_step）----
    return lam_vals_full, lam_vals, integral_step_full, integral_step, time_ll_pred, mask


# ============================================================
#  ETAS-PWL: 震级部分
# ============================================================

def etas_mag_input_pwl(T_dat, M_dat, Mcut, M0pred, params_3, time_step, Mmax):
    beta_3 = params_3['beta']

    Mmax = float(Mmax)
    Delta = Mmax - Mcut
    delta = M0pred - Mcut
    gamma = 1.5 * np.log(10.0)
    alpha = gamma - beta_3
    Z_norm = 1.0 - np.exp(-beta_3 * Delta)

    w_m_full = np.exp(gamma * (M_dat - M0pred))

    rate_vals_full = (beta_3 * np.exp(-beta_3 * (M_dat - Mcut))) / Z_norm
    intg_vals_full = (1 - np.exp(-beta_3 * (M_dat - Mcut))) / Z_norm

    mask = (M_dat >= M0pred)
    mags_mask = M_dat[mask]

    likelihood_truncated = w_m_full[mask] * (np.log(beta_3) - beta_3 * (mags_mask - M0pred) - np.log(1 - np.exp(-beta_3 * (Delta - delta))))

    cut = int(time_step) + 1

    count = sum(M_dat[: time_step + 1] >= M0pred)
    mag_ll_pred = likelihood_truncated[count:]

    rate_vals = rate_vals_full[cut:]
    intg_vals = intg_vals_full[cut:]
    assert np.all(rate_vals > 0), "Magnitude Rate should be positive! Something went wrong."
    return rate_vals_full, rate_vals, intg_vals_full, intg_vals, mag_ll_pred, mask
