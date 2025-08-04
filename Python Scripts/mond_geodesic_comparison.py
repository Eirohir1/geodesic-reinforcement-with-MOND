import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import glob

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
        raise ValueError(f"Expected at least 3 columns; got {arr.shape[1]}")
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

# CLASSIC MOND MODELS
def mond_simple_interpolation(r_kpc: np.ndarray, data: RotmodData, a0_kms2: float) -> np.ndarray:
    """Classic MOND with simple interpolation function"""
    
    v_bar = v_baryonic(r_kpc, data)
    v_bar_interp = np.interp(r_kpc, data.r_kpc, v_bar)
    
    # Convert to accelerations  
    G_kpc_solar = 4.3e-6  # G in units of kpc (km/s)^2 / M_solar
    
    # Estimate enclosed mass (rough approximation)
    M_enc = v_bar_interp**2 * r_kpc / G_kpc_solar  # Very rough
    a_N = G_kpc_solar * M_enc / r_kpc**2  # Newtonian acceleration
    
    # Simple interpolation function: μ(x) = x / (1 + x) where x = a_N/a0
    x = a_N / (a0_kms2 * 1e-10 * 3.086e16)  # Convert a0 to proper units
    mu = x / (1 + x)
    
    # MOND velocity
    v_mond = np.sqrt(mu * v_bar_interp**2)
    
    return v_mond

def mond_standard(r_kpc: np.ndarray, data: RotmodData, a0_kms2: float = 1.2e-10) -> np.ndarray:
    """Standard MOND with fixed a0"""
    return mond_simple_interpolation(r_kpc, data, a0_kms2)

# YOUR GEODESIC REINFORCEMENT
def geodesic_reinforcement(r_kpc: np.ndarray, data: RotmodData, 
                          alpha: float, ell_factor: float) -> np.ndarray:
    """Your proven geodesic reinforcement model"""
    v_bar = v_baryonic(r_kpc, data)
    v_bar_data = v_baryonic(data.r_kpc, data)
    v_bar_interp = np.interp(r_kpc, data.r_kpc, v_bar_data)
    
    R_galaxy = np.max(data.r_kpc)
    ell = ell_factor * R_galaxy
    
    dr = r_kpc[1] - r_kpc[0] if len(r_kpc) > 1 else 0.1
    r_conv = np.arange(0, r_kpc[-1] + 5*ell, dr)
    kern = np.exp(-r_conv / ell)
    v_bar_ext = np.interp(r_conv, r_kpc, v_bar_interp)
    
    conv_result = fftconvolve(v_bar_ext, kern, mode='same') * dr
    conv_interp = np.interp(r_kpc, r_conv, conv_result)
    
    v_dm = alpha * conv_interp
    v_total = np.sqrt(v_bar_interp**2 + np.maximum(v_dm, 0)**2)
    
    return v_total

# HYBRID MODELS
def mond_geodesic_additive(r_kpc: np.ndarray, data: RotmodData, 
                          a0_kms2: float, alpha: float, ell_factor: float) -> np.ndarray:
    """MOND + Geodesic (additive combination)"""
    
    v_mond = mond_simple_interpolation(r_kpc, data, a0_kms2)
    v_bar = v_baryonic(r_kpc, data)
    v_bar_interp = np.interp(r_kpc, data.r_kpc, v_bar)
    
    # Add geodesic enhancement
    R_galaxy = np.max(data.r_kpc)
    ell = ell_factor * R_galaxy
    
    dr = r_kpc[1] - r_kpc[0] if len(r_kpc) > 1 else 0.1
    r_conv = np.arange(0, r_kpc[-1] + 5*ell, dr)
    kern = np.exp(-r_conv / ell)
    v_bar_ext = np.interp(r_conv, r_kpc, v_bar_interp)
    
    conv_result = fftconvolve(v_bar_ext, kern, mode='same') * dr
    conv_interp = np.interp(r_kpc, r_conv, conv_result)
    
    v_geo = alpha * conv_interp
    
    # Combine: MOND provides base, geodesic provides enhancement
    v_total = np.sqrt(v_mond**2 + np.maximum(v_geo, 0)**2)
    
    return v_total

def mond_geodesic_kernel_enhanced(r_kpc: np.ndarray, data: RotmodData,
                                 a0_kms2: float, alpha: float, ell_factor: float) -> np.ndarray:
    """MOND with geodesic-enhanced interpolation function"""
    
    v_bar = v_baryonic(r_kpc, data)
    v_bar_data = v_baryonic(data.r_kpc, data)
    v_bar_interp = np.interp(r_kpc, data.r_kpc, v_bar_data)
    
    # Apply geodesic kernel to the baryonic matter first
    R_galaxy = np.max(data.r_kpc)
    ell = ell_factor * R_galaxy
    
    dr = r_kpc[1] - r_kpc[0] if len(r_kpc) > 1 else 0.1
    r_conv = np.arange(0, r_kpc[-1] + 5*ell, dr)
    kern = np.exp(-r_conv / ell)
    v_bar_ext = np.interp(r_conv, r_kpc, v_bar_interp)
    
    conv_result = fftconvolve(v_bar_ext, kern, mode='same') * dr
    conv_interp = np.interp(r_kpc, r_conv, conv_result)
    
    # This convolved field determines the effective acceleration
    G_kpc_solar = 4.3e-6
    M_eff = (alpha * conv_interp)**2 * r_kpc / G_kpc_solar
    a_eff = G_kpc_solar * M_eff / r_kpc**2
    
    # Use geodesic-enhanced acceleration in MOND formula
    x = a_eff / (a0_kms2 * 1e-10 * 3.086e16)
    mu = x / (1 + x)
    
    v_total = np.sqrt(mu * v_bar_interp**2 + (1-mu) * (alpha * conv_interp)**2)
    
    return v_total

def geodesic_derived_mond(r_kpc: np.ndarray, data: RotmodData, 
                         alpha: float, ell_factor: float) -> np.ndarray:
    """MOND-like behavior emerging FROM geodesic reinforcement"""
    
    v_bar = v_baryonic(r_kpc, data)
    v_bar_data = v_baryonic(data.r_kpc, data)
    v_bar_interp = np.interp(r_kpc, data.r_kpc, v_bar_data)
    
    # Geodesic convolution
    R_galaxy = np.max(data.r_kpc)
    ell = ell_factor * R_galaxy
    
    dr = r_kpc[1] - r_kpc[0] if len(r_kpc) > 1 else 0.1
    r_conv = np.arange(0, r_kpc[-1] + 5*ell, dr)
    kern = np.exp(-r_conv / ell)
    v_bar_ext = np.interp(r_conv, r_kpc, v_bar_interp)
    
    conv_result = fftconvolve(v_bar_ext, kern, mode='same') * dr
    conv_interp = np.interp(r_kpc, r_conv, conv_result)
    
    # Create MOND-like interpolation from geodesic structure
    ratio = conv_interp / np.maximum(v_bar_interp, 1e-6)
    
    # Natural interpolation emerges from convolution ratio
    mu_eff = ratio / (1 + ratio/alpha)
    
    v_total = np.sqrt(v_bar_interp**2 + mu_eff * (alpha * conv_interp)**2)
    
    return v_total

def fit_all_models(data: RotmodData):
    """Test all models: Pure MOND, Pure Geodesic, and Hybrids"""
    
    results = {}
    
    # 1. PURE GEODESIC (your proven model)
    def obj_geodesic(params):
        alpha, ell_factor = params
        if not (0.01 <= alpha <= 1.0) or not (0.1 <= ell_factor <= 2.0):
            return 1e8
        try:
            v_pred = geodesic_reinforcement(data.r_kpc, data, alpha, ell_factor)
            if np.any(~np.isfinite(v_pred)):
                return 1e8
            return np.sum(((data.v_obs - v_pred) / data.dv_obs)**2)
        except:
            return 1e8
    
    result = minimize(obj_geodesic, [0.3, 0.6], method='Nelder-Mead')
    if result.success:
        results['PURE_GEODESIC'] = {
            'params': result.x,
            'chi2': result.fun,
            'v_pred': geodesic_reinforcement(data.r_kpc, data, *result.x)
        }
    
    # 2. PURE MOND (standard)
    try:
        v_mond = mond_standard(data.r_kpc, data)
        chi2_mond = np.sum(((data.v_obs - v_mond) / data.dv_obs)**2)
        results['PURE_MOND'] = {
            'params': [1.2e-10],  # Standard a0
            'chi2': chi2_mond,
            'v_pred': v_mond
        }
    except:
        pass
    
    # 3. MOND + GEODESIC (additive)
    def obj_mond_geo_add(params):
        a0, alpha, ell_factor = params
        if not (0.5e-10 <= a0 <= 3e-10) or not (0.01 <= alpha <= 1.0) or not (0.1 <= ell_factor <= 2.0):
            return 1e8
        try:
            v_pred = mond_geodesic_additive(data.r_kpc, data, a0, alpha, ell_factor)
            if np.any(~np.isfinite(v_pred)):
                return 1e8
            return np.sum(((data.v_obs - v_pred) / data.dv_obs)**2)
        except:
            return 1e8
    
    result = minimize(obj_mond_geo_add, [1.2e-10, 0.2, 0.5], method='Nelder-Mead')
    if result.success:
        results['MOND_GEO_ADDITIVE'] = {
            'params': result.x,
            'chi2': result.fun,
            'v_pred': mond_geodesic_additive(data.r_kpc, data, *result.x)
        }
    
    # 4. GEODESIC-DERIVED MOND
    def obj_geo_mond(params):
        alpha, ell_factor = params
        if not (0.01 <= alpha <= 1.0) or not (0.1 <= ell_factor <= 2.0):
            return 1e8
        try:
            v_pred = geodesic_derived_mond(data.r_kpc, data, alpha, ell_factor)
            if np.any(~np.isfinite(v_pred)):
                return 1e8
            return np.sum(((data.v_obs - v_pred) / data.dv_obs)**2)
        except:
            return 1e8
    
    result = minimize(obj_geo_mond, [0.3, 0.6], method='Nelder-Mead')
    if result.success:
        results['GEODESIC_DERIVED_MOND'] = {
            'params': result.x,
            'chi2': result.fun,
            'v_pred': geodesic_derived_mond(data.r_kpc, data, *result.x)
        }
    
    return results

def mond_geodesic_collaboration():
    """Test if geodesic reinforcement can save MOND!"""
    
    print("🤝 MOND-GEODESIC COLLABORATION TEST")
    print("=" * 60)
    print("Can geodesic reinforcement provide the missing physics for MOND?")
    
    files = glob.glob("*_rotmod.dat")
    test_galaxies = files[:6]
    
    all_results = []
    
    for f in test_galaxies:
        print(f"\n🔬 Testing: {f}")
        try:
            data = read_rotmod(f)
            results = fit_all_models(data)
            
            print(f"Results for {f}:")
            for model_name, res in results.items():
                print(f"  {model_name}: χ² = {res['chi2']:.1f}")
            
            all_results.append((f, results))
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Analysis
    print(f"\n📊 COLLABORATION ANALYSIS:")
    print("=" * 40)
    
    model_names = ['PURE_GEODESIC', 'PURE_MOND', 'MOND_GEO_ADDITIVE', 'GEODESIC_DERIVED_MOND']
    
    for model in model_names:
        chi2_list = []
        for f, results in all_results:
            if model in results:
                chi2_list.append(results[model]['chi2'])
        
        if chi2_list:
            median_chi2 = np.median(chi2_list)
            print(f"{model}: median χ² = {median_chi2:.1f} ({len(chi2_list)} galaxies)")
    
    # Victory plot
    plt.figure(figsize=(18, 12))
    
    for i, (filename, results) in enumerate(all_results):
        if i >= 6:  # Limit plots
            break
            
        plt.subplot(2, 3, i+1)
        
        data = read_rotmod(filename)
        plt.errorbar(data.r_kpc, data.v_obs, data.dv_obs, 
                    fmt='ko', alpha=0.6, markersize=3, label='Observed')
        
        colors = ['red', 'blue', 'green', 'orange']
        linestyles = ['-', '--', '-.', ':']
        
        for j, (model, color, ls) in enumerate(zip(model_names, colors, linestyles)):
            if model in results:
                plt.plot(data.r_kpc, results[model]['v_pred'], 
                        color=color, linestyle=ls, linewidth=2,
                        label=f"{model.replace('_', ' ')} (χ²={results[model]['chi2']:.1f})")
        
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        plt.title(f'{filename.replace("_rotmod.dat", "")}')
        plt.legend(fontsize=6)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('MOND-GEODESIC COLLABORATION: Saving MOND with Physics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mond_geodesic_collaboration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return all_results

if __name__ == "__main__":
    results = mond_geodesic_collaboration()
    
    print(f"\n🤝 Can geodesic reinforcement save MOND?")
    print(f"The answer will revolutionize dark matter physics!")