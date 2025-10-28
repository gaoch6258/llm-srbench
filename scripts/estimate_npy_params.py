#!/usr/bin/env python3
"""
Estimate dx, dy, dt and parameter ranges for the fixed chemotaxis PDE

PDE: ∂g/∂t = D·Δg - χ·∇·(g·∇ln(S)) + r·g·(1-g/K)

This script loads S and g from npy files (shape (T, H, W)), standardizes by
dividing by std immediately after loading, performs a coarse-to-fine search
over dx, dy (=dx), and dt across multiple orders of magnitude, fits PDE
parameters at each grid point via multi-start optimization on a dg/dt loss,
selects the best (dx,dy,dt) by loss, and reports parameter ranges.

Also produces a 9-panel visualization (6 rollout frames + 3 statistics) using
the best (dx,dy,dt) and parameters.
"""

import argparse
from pathlib import Path
import json
import time
import numpy as np
from typing import Tuple, Dict, List

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from bench.pde_visualization import PDEVisualizer


def load_and_standardize(ca_path: str, cell_path: str, crop_size: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Load S (ca) and g (cell) from npy (T,H,W); return standardized (H,W,T).

    S and g are divided by their own std immediately after load. Numerical
    clamp is applied to S to keep it positive for ln(S).
    """
    S_np = np.load(ca_path)
    g_np = np.load(cell_path)
    assert S_np.ndim == 3 and g_np.ndim == 3, f"Expected (T,H,W); got S{S_np.shape}, g{g_np.shape}"

    # Convert to (H,W,T)
    S = np.transpose(S_np, (1, 2, 0)).astype(np.float32)
    g = np.transpose(g_np, (1, 2, 0)).astype(np.float32)

    # Standardize by std only (per user request)
    S_std = float(S.std())
    g_std = float(g.std())
    S = S / max(S_std, 1e-8)
    g = g / max(g_std, 1e-8)

    # Ensure positivity for ln(S)
    S = np.maximum(S, 1e-6)

    # Optional center crop to square crop_size
    if crop_size and crop_size > 0:
        H, W, _ = g.shape
        ch = min(H, crop_size)
        cw = min(W, crop_size)
        y0 = (H - ch) // 2
        x0 = (W - cw) // 2
        g = g[y0:y0+ch, x0:x0+cw, :]
        S = S[y0:y0+ch, x0:x0+cw, :]

    return S, g


def laplacian_xy(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute spatial Laplacian along x,y only for a 3D array (H,W,T-1)."""
    d2x = np.gradient(np.gradient(a, dx, axis=1), dx, axis=1)
    d2y = np.gradient(np.gradient(a, dy, axis=0), dy, axis=0)
    return d2x + d2y


def predict_dgdt(g_series: np.ndarray, S_series: np.ndarray, dx: float, dy: float,
                 params: np.ndarray) -> np.ndarray:
    """Vectorized prediction of dg/dt across time t=0..T-2.

    Shapes: g_series, S_series are (H,W,T); returns (H,W,T-1)
    """
    D, chi, r, K = params
    g_t = g_series[:, :, :-1]
    S_t = S_series[:, :, :-1]

    # ln(S) and gradients
    lnS = np.log(np.maximum(S_t, 1e-6))
    grad_lnS_x = np.gradient(lnS, dx, axis=1)
    grad_lnS_y = np.gradient(lnS, dy, axis=0)

    # Flux and divergence
    flux_x = g_t * grad_lnS_x
    flux_y = g_t * grad_lnS_y
    div_flux = np.gradient(flux_x, dx, axis=1) + np.gradient(flux_y, dy, axis=0)

    # Diffusion and reaction
    lap_g = laplacian_xy(g_t, dx, dy)
    reaction = r * g_t * (1.0 - g_t / (K + 1e-8))
    return D * lap_g - chi * div_flux + reaction


def one_step(g2d: np.ndarray, S2d: np.ndarray, dx: float, dy: float, dt: float, params: np.ndarray) -> np.ndarray:
    """One forward-Euler step for the PDE on 2D slices."""
    D, chi, r, K = params
    # Spatial operators using np.gradient for consistency with vectorized path
    d2x = np.gradient(np.gradient(g2d, dx, axis=1), dx, axis=1)
    d2y = np.gradient(np.gradient(g2d, dy, axis=0), dy, axis=0)
    lap = d2x + d2y
    lnS = np.log(np.maximum(S2d, 1e-6))
    gx = np.gradient(lnS, dx, axis=1)
    gy = np.gradient(lnS, dy, axis=0)
    div = np.gradient(g2d * gx, dx, axis=1) + np.gradient(g2d * gy, dy, axis=0)
    react = r * g2d * (1.0 - g2d / (K + 1e-8))
    dgdt = D * lap - chi * div + react
    return np.maximum(g2d + dt * dgdt, 0.0)


def fit_params_for_grid(g: np.ndarray, S: np.ndarray, dx: float, dy: float, dt: float,
                        restarts: int, maxiter: int, rng: np.random.Generator,
                        bounds: List[Tuple[float, float]]) -> Dict:
    """Fit D, chi, r, K at fixed dx,dy,dt using dg/dt MSE objective with multi-start."""
    from scipy.optimize import minimize

    dgdt_obs = (g[:, :, 1:] - g[:, :, :-1]) / dt

    def objective(x):
        D, chi, r, K = x
        if D < 0 or chi < 0 or r < 0 or K <= 0:
            return 1e12
        pred = predict_dgdt(g, S, dx, dy, x)
        # Mean squared error
        return float(np.mean((pred - dgdt_obs) ** 2))

    trials = []
    best = None
    for i in range(restarts):
        x0 = np.array([
            rng.uniform(*bounds[0]),
            rng.uniform(*bounds[1]),
            rng.uniform(*bounds[2]),
            rng.uniform(*bounds[3]),
        ], dtype=np.float64)
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={"maxiter": int(maxiter)})
        trials.append((res.x.copy(), float(res.fun)))
        if best is None or res.fun < best[1]:
            best = (res.x.copy(), float(res.fun))
    params_arr = np.array([t[0] for t in trials])
    losses = np.array([t[1] for t in trials])
    return {
        'best_params': best[0],
        'best_loss': best[1],
        'all_params': params_arr,
        'all_losses': losses,
    }


def main():
    ap = argparse.ArgumentParser(description='Estimate dx, dy, dt and parameter ranges for npy data.')
    ap.add_argument('--dataset', type=str, default='data', help='Folder containing ca_video_continuous.npy and cell_video_continuous.npy')
    ap.add_argument('--ca_path', type=str, default=None, help='Override path to S npy (shape T,H,W)')
    ap.add_argument('--cell_path', type=str, default=None, help='Override path to g npy (shape T,H,W)')
    ap.add_argument('--out_dir', type=str, default='logs/npy_param_estimation', help='Output directory')
    ap.add_argument('--crop_size', type=int, default=256, help='Center crop size to speed up (0 to disable)')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed')

    # Grid for dx=dy and dt (logspace ranges to span orders of magnitude)
    ap.add_argument('--dx_min', type=float, default=0.25)
    ap.add_argument('--dx_max', type=float, default=4.0)
    ap.add_argument('--dx_num', type=int, default=5)
    ap.add_argument('--dt_min', type=float, default=0.1)
    ap.add_argument('--dt_max', type=float, default=10.0)
    ap.add_argument('--dt_num', type=int, default=7)

    # Optimization settings
    ap.add_argument('--restarts', type=int, default=8)
    ap.add_argument('--maxiter', type=int, default=250)

    args = ap.parse_args()

    out_dir = Path(args.out_dir) / time.strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve file paths
    if args.ca_path is None or args.cell_path is None:
        folder = Path(args.dataset)
        ca_path = folder / 'ca_video_continuous.npy'
        cell_path = folder / 'cell_video_continuous.npy'
    else:
        ca_path = Path(args.ca_path)
        cell_path = Path(args.cell_path)

    # Load and standardize
    S, g = load_and_standardize(str(ca_path), str(cell_path), crop_size=args.crop_size)
    H, W, T = g.shape
    print(f"Loaded and standardized: g,S shapes = {g.shape}, {S.shape}")

    rng = np.random.default_rng(args.seed)

    # Build log-spaced grids
    dx_candidates = np.geomspace(args.dx_min, args.dx_max, num=args.dx_num)
    dt_candidates = np.geomspace(args.dt_min, args.dt_max, num=args.dt_num)

    # Bounds for standardized data (g scaled by std): K around O(1)
    g_max = float(np.max(g))
    bounds = [
        (1e-6, 1e3),   # D
        (1e-6, 1e3),   # chi
        (1e-6, 10.0),  # r
        (max(1e-2, 0.1 * g_max), max(20.0, 10.0 * g_max)),  # K
    ]

    grid_results = []
    best_global = None
    total = len(dx_candidates) * len(dt_candidates)
    idx = 0
    for dx in dx_candidates:
        dy = dx
        for dt in dt_candidates:
            idx += 1
            print(f"[{idx}/{total}] Fitting for dx=dy={dx:.4g}, dt={dt:.4g} ...")
            res = fit_params_for_grid(g, S, float(dx), float(dy), float(dt), args.restarts, args.maxiter, rng, bounds)
            grid_results.append({
                'dx': float(dx), 'dy': float(dy), 'dt': float(dt),
                'best_params': res['best_params'].tolist(),
                'best_loss': float(res['best_loss']),
                'all_params': res['all_params'].tolist(),
                'all_losses': res['all_losses'].tolist(),
            })
            if best_global is None or res['best_loss'] < best_global['best_loss']:
                best_global = {
                    'dx': float(dx), 'dy': float(dy), 'dt': float(dt),
                    'best_params': res['best_params'].copy(),
                    'best_loss': float(res['best_loss']),
                    'all_params': res['all_params'].copy(),
                    'all_losses': res['all_losses'].copy(),
                }

    # Save raw grid results
    with open(out_dir / 'grid_results.json', 'w') as f:
        json.dump(grid_results, f)

    # Summarize best cell and param ranges at that cell
    # Compute descriptive stats for D,chi,r,K across restarts at best cell
    arr = best_global['all_params']
    D_list = arr[:, 0]
    chi_list = arr[:, 1]
    r_list = arr[:, 2]
    K_list = arr[:, 3]
    ranges = {
        'D': {'min': float(D_list.min()), 'max': float(D_list.max()), 'mean': float(D_list.mean()), 'median': float(np.median(D_list))},
        'chi': {'min': float(chi_list.min()), 'max': float(chi_list.max()), 'mean': float(chi_list.mean()), 'median': float(np.median(chi_list))},
        'r': {'min': float(r_list.min()), 'max': float(r_list.max()), 'mean': float(r_list.mean()), 'median': float(np.median(r_list))},
        'K': {'min': float(K_list.min()), 'max': float(K_list.max()), 'mean': float(K_list.mean()), 'median': float(np.median(K_list))},
    }

    best_summary = {
        'best_dx': best_global['dx'], 'best_dy': best_global['dy'], 'best_dt': best_global['dt'],
        'best_params': {'D': float(best_global['best_params'][0]), 'chi': float(best_global['best_params'][1]), 'r': float(best_global['best_params'][2]), 'K': float(best_global['best_params'][3])},
        'best_loss': float(best_global['best_loss']),
        'ranges': ranges,
    }
    with open(out_dir / 'best_summary.json', 'w') as f:
        json.dump(best_summary, f, indent=2)

    print("\nBest grid cell:")
    print(json.dumps(best_summary, indent=2))

    # Rollout with the best settings and plot 9-panel
    dx = float(best_global['dx'])
    dy = float(best_global['dy'])
    dt = float(best_global['dt'])
    params = np.asarray(best_global['best_params'], dtype=np.float32)

    Tn = g.shape[2]
    g_roll = np.zeros_like(g)
    g_roll[:, :, 0] = g[:, :, 0]
    for t in range(1, Tn):
        S_t = S[:, :, t]
        g_roll[:, :, t] = one_step(g_roll[:, :, t-1], S_t, dx, dy, dt, params)

    viz_path = out_dir / 'rollout_6x3.png'
    PDEVisualizer().create_rollout_grid_with_stats(observed=g, predicted=g_roll, save_path=str(viz_path))
    print(f"Saved rollout and stats visualization to: {viz_path}")


if __name__ == '__main__':
    main()
