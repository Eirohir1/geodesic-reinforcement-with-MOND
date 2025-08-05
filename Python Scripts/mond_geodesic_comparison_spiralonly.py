import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import os

# === SET TO YOUR TRUE DATA DIR ===
DATA_DIR = r'C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts'

spiral_galaxies = [
    'CamB_rotmod.dat', 'D512-2_rotmod.dat', 'D564-8_rotmod.dat', 'D631-7_rotmod.dat',
    'DDO064_rotmod.dat', 'DDO154_rotmod.dat', 'DDO161_rotmod.dat', 'DDO168_rotmod.dat',
    'DDO170_rotmod.dat', 'ESO079-G014_rotmod.dat', 'ESO116-G012_rotmod.dat',
    'ESO444-G084_rotmod.dat', 'ESO563-G021_rotmod.dat', 'F561-1_rotmod.dat',
    'F563-1_rotmod.dat', 'F563-V1_rotmod.dat', 'F563-V2_rotmod.dat', 'F565-V2_rotmod.dat',
    'F567-2_rotmod.dat', 'F568-1_rotmod.dat'
]

print(f"\nChecking DATA_DIR: {DATA_DIR}")
if os.path.exists(DATA_DIR):
    files_in_dir = os.listdir(DATA_DIR)
    print(f"Directory exists. Files inside:")
    for f in files_in_dir:
        print(" ", f)
else:
    print("ERROR: DATA_DIR does not exist! Double-check your path.")
    exit(1)

print("\nChecking each spiral galaxy file in spiral_galaxies list:")
good_files = []
for fname in spiral_galaxies:
    path = os.path.join(DATA_DIR, fname)
    exists = os.path.exists(path)
    print(f"  {fname}: {'FOUND' if exists else 'MISSING'}")
    if exists:
        good_files.append(path)

@dataclass
class RotmodData:
    r_kpc: np.ndarray
    v_obs: np.ndarray
    dv_obs: np.ndarray
    v_gas: Optional[np.ndarray] = None
    v_disk: Optional[np.ndarray] = None
    v_bulge: Optional[np.ndarray] = None

def read_rotmod(path: str) -> RotmodData:
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(";"):
                continue
            parts = s.replace(",", " ").split()
            vals = []
            ok = True
            for x in parts:
                try:
                    vals.append(float(x))
                except ValueError:
                    ok = False
                    break
            if ok and len(vals) >= 3:
                rows.append(vals)
    if not rows:
        raise ValueError(f"No valid numeric rows in {path}")
    arr = np.array(rows, dtype=float)
    r = arr[:, 0]
    vobs = arr[:, 1] 
    dv = arr[:, 2]
    vgas = arr[:, 3] if arr.shape[1] > 3 else None
    vdisk = arr[:, 4] if arr.shape[1] > 4 else None
    vbulge = arr[:, 5] if arr.shape[1] > 5 else None
    if np.any(r <= 0):
        raise ValueError("Non-positive radii detected")
    if np.any(dv <= 0):
        raise ValueError("Non-positive velocity errors detected")
    return RotmodData(r, vobs, dv, vgas, vdisk, vbulge)

def v_baryonic(r_kpc: np.ndarray, data: RotmodData) -> np.ndarray:
    v_bar_squared = np.zeros_like(r_kpc)
    if data.v_gas is not None:
        v_gas_interp = np.interp(r_kpc, data.r_kpc, data.v_gas)
        v_bar_squared += np.maximum(v_gas_interp, 0)**2
    if data.v_disk is not None:
        v_disk_interp = np.interp(r_kpc, data.r_kpc, data.v_disk) 
        v_bar_squared += np.maximum(v_disk_interp, 0)**2
    if data.v_bulge is not None:
        v_bulge_interp = np.interp(r_kpc, data.r_kpc, data.v_bulge)
        v_bar_squared += np.maximum(v_bulge_interp, 0)**2
    return np.sqrt(v_bar_squared)

def mond_standard(r_kpc: np.ndarray, data: RotmodData, a0_kms2: float = 1.2e-10) -> np.ndarray:
    v_bar = v_baryonic(r_kpc, data)
    mask = r_kpc > 0
    a_N = np.zeros_like(r_kpc)
    a_N[mask] = v_bar[mask]**2 / r_kpc[mask]
    a_N_SI = a_N * 1e6 / 3.086e19
    x = a_N_SI / a0_kms2
    mu = x / (1 + x)
    v_mond = np.zeros_like(v_bar)
    mu_safe = np.maximum(mu, 1e-10)
    v_mond = v_bar / np.sqrt(mu_safe)
    return v_mond

def geodesic_reinforcement(r_kpc: np.ndarray, data: RotmodData, alpha: float, ell_factor: float) -> np.ndarray:
    v_bar = v_baryonic(r_kpc, data)
    if len(r_kpc) > 1:
        dr = np.median(np.diff(r_kpc))
    else:
        dr = 0.1
    R_galaxy = np.max(data.r_kpc)
    ell = ell_factor * R_galaxy
    r_max = max(r_kpc[-1], R_galaxy) + 5*ell
    r_conv = np.arange(0, r_max, dr)
    kern = np.exp(-r_conv / ell)
    kern = kern / np.trapz(kern, r_conv)
    v_bar_ext = np.interp(r_conv, r_kpc, v_bar)
    conv_result = fftconvolve(v_bar_ext, kern, mode='same') * dr
    conv_interp = np.interp(r_kpc, r_conv, conv_result)
    v_dm = alpha * conv_interp
    v_total = np.sqrt(v_bar**2 + np.maximum(v_dm, 0)**2)
    return v_total

def fit_model_robust(data: RotmodData, model_func, param_bounds, initial_guess):
    def objective(params):
        try:
            for i, (low, high) in enumerate(param_bounds):
                if not (low <= params[i] <= high):
                    return 1e8
            v_pred = model_func(data.r_kpc, data, *params)
            if np.any(~np.isfinite(v_pred)) or np.any(v_pred <= 0):
                return 1e8
            residuals = (data.v_obs - v_pred) / data.dv_obs
            chi2 = np.sum(residuals**2)
            return chi2
        except Exception:
            return 1e8
    methods = ['Nelder-Mead', 'Powell', 'BFGS']
    best_result = None
    best_chi2 = 1e8
    for method in methods:
        try:
            result = minimize(objective, initial_guess, method=method,
                            options={'maxiter': 1000})
            if result.success and result.fun < best_chi2:
                best_result = result
                best_chi2 = result.fun
        except:
            continue
    return best_result

def compare_all_models(data: RotmodData):
    results = {}
    result_geo = fit_model_robust(
        data, geodesic_reinforcement,
        [(0.01, 1.0), (0.1, 2.0)],
        [0.3, 0.6]
    )
    if result_geo and result_geo.success:
        alpha_opt, ell_opt = result_geo.x
        v_pred_geo = geodesic_reinforcement(data.r_kpc, data, alpha_opt, ell_opt)
        results['GEODESIC'] = {
            'params': result_geo.x,
            'chi2': result_geo.fun,
            'v_pred': v_pred_geo,
            'dof': len(data.r_kpc) - 2,
            'reduced_chi2': result_geo.fun / (len(data.r_kpc) - 2)
        }
    try:
        v_pred_mond = mond_standard(data.r_kpc, data)
        chi2_mond = np.sum(((data.v_obs - v_pred_mond) / data.dv_obs)**2)
        results['MOND'] = {
            'params': [1.2e-10],
            'chi2': chi2_mond,
            'v_pred': v_pred_mond,
            'dof': len(data.r_kpc) - 1,
            'reduced_chi2': chi2_mond / (len(data.r_kpc) - 1)
        }
    except Exception as e:
        print(f"MOND failed: {e}")
    return results

def rigorous_comparison():
    print("\n🔬 RIGOROUS MOND VS GEODESIC COMPARISON (Spiral sample)")
    print("=" * 60)
    if not good_files:
        print("No spiral galaxy files found in this directory!")
        return
    all_results = []
    for filename in good_files:
        print(f"\nAnalyzing: {filename}")
        try:
            data = read_rotmod(filename)
            results = compare_all_models(data)
            if results:
                print(f"  Results:")
                for model, res in results.items():
                    print(f"    {model}: χ²/dof = {res['reduced_chi2']:.2f}")
                all_results.append((filename, results))
            else:
                print(f"  No successful fits")
        except Exception as e:
            print(f"  Error: {e}")
    if all_results:
        print(f"\n📊 STATISTICAL SUMMARY ({len(all_results)} spiral galaxies):")
        print("=" * 50)
        for model_name in ['GEODESIC', 'MOND']:
            chi2_values = []
            for filename, results in all_results:
                if model_name in results:
                    chi2_values.append(results[model_name]['reduced_chi2'])
            if chi2_values:
                chi2_array = np.array(chi2_values)
                print(f"{model_name}:")
                print(f"  Median χ²/dof: {np.median(chi2_array):.2f}")
                print(f"  Mean χ²/dof: {np.mean(chi2_array):.2f}")
                print(f"  Std χ²/dof: {np.std(chi2_array):.2f}")
                print(f"  Success rate: {len(chi2_values)}/{len(all_results)} = {100*len(chi2_values)/len(all_results):.1f}%")
    return all_results

if __name__ == "__main__":
    rigorous_comparison()
