import os, glob
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, List
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

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
            if ok and vals:
                rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric rows parsed in {path}")
    arr = np.array(rows, dtype=float)
    if arr.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns (r, v_obs, dv); got {arr.shape[1]}")
    r = arr[:, 0]; vobs = arr[:, 1]; dv = arr[:, 2]
    vgas = arr[:, 3] if arr.shape[1] > 3 else None
    vdisk = arr[:, 4] if arr.shape[1] > 4 else None
    vbulge = arr[:, 5] if arr.shape[1] > 5 else None
    return RotmodData(r, vobs, dv, vgas, vdisk, vbulge)

def v_baryonic(r_kpc: np.ndarray, data: RotmodData) -> np.ndarray:
    v_bar = np.zeros_like(r_kpc)
    if data.v_gas is not None:
        v_bar += np.interp(r_kpc, data.r_kpc, data.v_gas)**2
    if data.v_disk is not None:
        v_bar += np.interp(r_kpc, data.r_kpc, data.v_disk)**2
    if data.v_bulge is not None:
        v_bar += np.interp(r_kpc, data.r_kpc, data.v_bulge)**2
    return np.sqrt(v_bar)

def reinf_kernel(r: np.ndarray, ell: float) -> np.ndarray:
    return np.exp(-r / ell)

def v_dark(r_kpc: np.ndarray, data: RotmodData, ell: float, 
          alpha: float, g_inf: float) -> np.ndarray:
    v_bar = v_baryonic(r_kpc, data)
    v_bar_data = v_baryonic(data.r_kpc, data)
    v_bar_interp = np.interp(r_kpc, data.r_kpc, v_bar_data)
    
    # Convolution
    dr = r_kpc[1] - r_kpc[0] if len(r_kpc) > 1 else 0.1
    r_conv = np.arange(0, r_kpc[-1] + 5*ell, dr)
    kern = reinf_kernel(r_conv, ell)
    v_bar_ext = np.interp(r_conv, r_kpc, v_bar_interp)
    
    # Geodesic reinforcement
    conv_result = fftconvolve(v_bar_ext, kern, mode='same') * dr
    conv_interp = np.interp(r_kpc, r_conv, conv_result)
    
    # Dark matter velocity
    v_dm = alpha * conv_interp + g_inf
    return np.maximum(v_dm, 0)

def v_total(r_kpc: np.ndarray, data: RotmodData, 
           ell: float, alpha: float, g_inf: float) -> np.ndarray:
    v_bar = v_baryonic(r_kpc, data)
    v_dm = v_dark(r_kpc, data, ell, alpha, g_inf)
    return np.sqrt(v_bar**2 + v_dm**2)

def fit_galaxy_adaptive(data: RotmodData):
    """Adaptive fitting based on galaxy properties"""
    
    # Analyze galaxy
    peak_v = np.max(data.v_obs)
    r_max = np.max(data.r_kpc)
    v_bar_data = v_baryonic(data.r_kpc, data)
    peak_v_bar = np.max(v_bar_data)
    bar_frac = peak_v_bar/peak_v if peak_v > 0 else 0
    n_points = len(data.r_kpc)
    
    # Adaptive parameter ranges
    if peak_v < 50:  # Dwarf
        ell_range = (0.5, 10.0)
        alpha_range = (0.05, 0.6) 
        g_inf_range = (0.0, 0.4)
        chi2_threshold = 500
    elif bar_frac > 0.75:  # Baryonic-dominated
        ell_range = (2.0, 25.0)
        alpha_range = (0.01, 0.4)
        g_inf_range = (0.0, 0.2)
        chi2_threshold = max(1000, n_points * 100)  # Scale with complexity
    else:  # DM-dominated
        ell_range = (3.0, 35.0)
        alpha_range = (0.05, 1.0)
        g_inf_range = (0.0, 0.6)
        chi2_threshold = max(2000, n_points * 50)
    
    def objective(params):
        ell, alpha, g_inf = params
        
        if not (ell_range[0] <= ell <= ell_range[1]): return 1e8
        if not (alpha_range[0] <= alpha <= alpha_range[1]): return 1e8  
        if not (g_inf_range[0] <= g_inf <= g_inf_range[1]): return 1e8
        
        try:
            v_pred = v_total(data.r_kpc, data, ell, alpha, g_inf)
            
            # Prevent NaN/inf
            if np.any(~np.isfinite(v_pred)):
                return 1e8
                
            chi2 = np.sum(((data.v_obs - v_pred) / data.dv_obs)**2)
            return chi2
        except:
            return 1e8
    
    # Smart starting points
    attempts = [
        (np.mean(ell_range), np.mean(alpha_range), np.mean(g_inf_range)),
        (r_max/3, 0.15, 0.05),
        (2.0, 0.3, 0.1),
        (ell_range[0]*1.5, alpha_range[1]*0.7, g_inf_range[0]),
        (ell_range[1]*0.7, alpha_range[0]*2, g_inf_range[1]*0.5)
    ]
    
    best_params = None
    best_chi2 = 1e8
    
    for start in attempts:
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'fatol': 1e-6})
            if result.fun < best_chi2:
                best_chi2 = result.fun
                best_params = result.x
        except:
            continue
    
    if best_params is not None and best_chi2 < chi2_threshold:
        return best_params, best_chi2
    else:
        return None, best_chi2

def analyze_correlations(results):
    """Analyze parameter correlations"""
    
    ells = np.array([r['ell'] for r in results])
    alphas = np.array([r['alpha'] for r in results])
    g_infs = np.array([r['g_inf'] for r in results])
    chi2s = np.array([r['chi2'] for r in results])
    r_maxes = np.array([r['r_max'] for r in results])
    v_maxes = np.array([r['v_max'] for r in results])
    
    print(f"\n📈 PARAMETER STATISTICS:")
    print(f"ell: {ells.min():.1f} - {ells.max():.1f} kpc (median={np.median(ells):.1f})")
    print(f"α: {alphas.min():.3f} - {alphas.max():.3f} (median={np.median(alphas):.3f})")
    print(f"g∞: {g_infs.min():.3f} - {g_infs.max():.3f} (median={np.median(g_infs):.3f})")
    print(f"χ²: {np.median(chi2s):.1f} (median)")
    
    # Key correlations
    corr_ell_size = np.corrcoef(ells, r_maxes)[0,1]
    corr_alpha_vel = np.corrcoef(alphas, v_maxes)[0,1]
    
    print(f"\n🔬 KEY CORRELATIONS:")
    print(f"ell vs galaxy_size: r = {corr_ell_size:.3f}")
    print(f"α vs peak_velocity: r = {corr_alpha_vel:.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].scatter(r_maxes, ells, alpha=0.6, s=30)
    axes[0,0].set_xlabel('Galaxy Size (kpc)')
    axes[0,0].set_ylabel('ell parameter (kpc)')
    axes[0,0].set_title(f'ell vs Size (r={corr_ell_size:.3f})')
    
    axes[0,1].scatter(v_maxes, alphas, alpha=0.6, s=30) 
    axes[0,1].set_xlabel('Peak Velocity (km/s)')
    axes[0,1].set_ylabel('α parameter')
    axes[0,1].set_title(f'α vs Velocity (r={corr_alpha_vel:.3f})')
    
    axes[0,2].hist(chi2s, bins=30, alpha=0.7)
    axes[0,2].set_xlabel('χ² values')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title('Fit Quality')
    axes[0,2].axvline(np.median(chi2s), color='red', linestyle='--')
    
    axes[1,0].hist(ells, bins=20, alpha=0.7, color='green')
    axes[1,0].set_xlabel('ell (kpc)')
    axes[1,0].set_title('ell Distribution')
    
    axes[1,1].hist(alphas, bins=20, alpha=0.7, color='orange')
    axes[1,1].set_xlabel('α parameter') 
    axes[1,1].set_title('α Distribution')
    
    axes[1,2].hist(g_infs, bins=20, alpha=0.7, color='purple')
    axes[1,2].set_xlabel('g∞ parameter')
    axes[1,2].set_title('g∞ Distribution')
    
    plt.tight_layout()
    plt.savefig('sparc_full_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def full_sparc_validation_with_failures():
    """Run validation and analyze failures"""
    print("=== FULL SPARC VALIDATION WITH FAILURE ANALYSIS ===")
    
    files = glob.glob("*.dat")
    rotmod_files = [f for f in files if f.lower().endswith("_rotmod.dat")]
    
    print(f"Found {len(rotmod_files)} SPARC galaxies")
    
    results = []
    failed_details = []
    
    for i, f in enumerate(rotmod_files):
        if i % 25 == 0:
            print(f"Progress: {i+1}/{len(rotmod_files)}")
            
        try:
            name = f.replace("_rotmod.dat", "")
            data = read_rotmod(f)
            params, chi2 = fit_galaxy_adaptive(data)
            
            # Get galaxy properties regardless of fit success
            peak_v = np.max(data.v_obs)
            r_max = np.max(data.r_kpc)
            v_bar_data = v_baryonic(data.r_kpc, data)
            peak_v_bar = np.max(v_bar_data)
            bar_frac = peak_v_bar/peak_v if peak_v > 0 else 0
            n_points = len(data.r_kpc)
            
            if params is not None:
                results.append({
                    'name': name,
                    'ell': params[0],
                    'alpha': params[1],
                    'g_inf': params[2], 
                    'chi2': chi2,
                    'r_max': r_max,
                    'v_max': peak_v,
                    'v_bar_max': peak_v_bar,
                    'n_points': n_points
                })
            else:
                # Classify failed galaxy
                if 'NGC' in name and peak_v > 100:
                    gal_type = "LARGE_SPIRAL"
                elif 'NGC' in name:
                    gal_type = "SPIRAL/ELLIPTICAL"
                elif any(x in name for x in ['DDO', 'UGC', 'UGCA']):
                    if peak_v < 30:
                        gal_type = "ULTRA_DWARF"
                    elif peak_v < 60:
                        gal_type = "DWARF"
                    else:
                        gal_type = "INTERMEDIATE"
                elif name.startswith('F'):
                    gal_type = "FIELD_GALAXY"
                elif name.startswith('IC'):
                    gal_type = "IC_GALAXY"
                else:
                    gal_type = "UNKNOWN"
                
                # Determine failure reason
                if n_points < 8:
                    reason = "TOO_FEW_POINTS"
                elif bar_frac > 0.95:
                    reason = "PURELY_BARYONIC"
                elif peak_v < 15:
                    reason = "TOO_SLOW"
                elif peak_v > 300:
                    reason = "TOO_FAST"
                elif r_max < 1.0:
                    reason = "TOO_COMPACT"
                else:
                    reason = "COMPLEX_STRUCTURE"
                
                failed_details.append({
                    'name': name,
                    'type': gal_type,
                    'reason': reason,
                    'peak_v': peak_v,
                    'r_max': r_max,
                    'bar_frac': bar_frac,
                    'n_points': n_points,
                    'chi2_attempted': chi2
                })
                
        except Exception as e:
            name = f.replace("_rotmod.dat", "")
            failed_details.append({
                'name': name,
                'type': "CRASH",
                'reason': "DATA_ERROR",
                'peak_v': 0,
                'r_max': 0,
                'bar_frac': 0,
                'n_points': 0,
                'chi2_attempted': 1e6
            })
    
    print(f"\n📊 RESULTS:")
    print(f"✅ Successful fits: {len(results)}")
    print(f"❌ Failed fits: {len(failed_details)}")
    print(f"🎯 Success rate: {100*len(results)/(len(results)+len(failed_details)):.1f}%")
    
    if len(results) > 0:
        analyze_correlations(results)
    
    # Print detailed failure report
    print(f"\n📋 DETAILED FAILURE REPORT:")
    print(f"{'Name':<15} {'Type':<15} {'Reason':<18} {'V_peak':<8} {'R_max':<8} {'Bar_frac':<8} {'N_pts':<6}")
    print("-" * 90)
    
    for f in failed_details:
        print(f"{f['name']:<15} {f['type']:<15} {f['reason']:<18} {f['peak_v']:<8.1f} {f['r_max']:<8.1f} {f['bar_frac']:<8.2f} {f['n_points']:<6}")
    
    # Categorize failures
    failure_types = {}
    failure_reasons = {}
    
    for f in failed_details:
        failure_types[f['type']] = failure_types.get(f['type'], 0) + 1
        failure_reasons[f['reason']] = failure_reasons.get(f['reason'], 0) + 1
    
    print(f"\n📊 FAILURE BY GALAXY TYPE:")
    for gtype, count in failure_types.items():
        print(f"   {gtype}: {count}")
    
    print(f"\n📊 FAILURE BY REASON:")
    for reason, count in failure_reasons.items():
        print(f"   {reason}: {count}")
    
    return results, failed_details

if __name__ == "__main__":
    results, failures = full_sparc_validation_with_failures()