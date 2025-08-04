import os, glob
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from scipy import stats
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

# MULTIPLE KERNEL ARSENAL
def exponential_kernel(r: np.ndarray, ell: float) -> np.ndarray:
    """Your original - exponential decay"""
    return np.exp(-r / ell)

def gaussian_kernel(r: np.ndarray, ell: float) -> np.ndarray:
    """Gaussian - common in physics"""
    return np.exp(-0.5 * (r / ell)**2)

def power_law_kernel(r: np.ndarray, ell: float) -> np.ndarray:
    """Power law - scale invariant"""
    return (1 + r / ell)**(-2)

def yukawa_kernel(r: np.ndarray, ell: float) -> np.ndarray:
    """Yukawa - massive field theory"""
    return np.exp(-r / ell) / (1 + r / ell)

def lorentzian_kernel(r: np.ndarray, ell: float) -> np.ndarray:
    """Lorentzian - common in relativity"""
    return 1 / (1 + (r / ell)**2)

def step_kernel(r: np.ndarray, ell: float) -> np.ndarray:
    """Step function - completely unphysical"""
    return np.where(r < ell, 1.0, 0.0)

# GENERALIZED DARK MATTER FUNCTION
def v_dark_general(r_kpc: np.ndarray, data: RotmodData, ell: float, 
                  alpha: float, g_inf: float, kernel_func) -> np.ndarray:
    """Generalized dark matter with arbitrary kernel"""
    v_bar = v_baryonic(r_kpc, data)
    v_bar_data = v_baryonic(data.r_kpc, data)
    v_bar_interp = np.interp(r_kpc, data.r_kpc, v_bar_data)
    
    dr = r_kpc[1] - r_kpc[0] if len(r_kpc) > 1 else 0.1
    r_conv = np.arange(0, r_kpc[-1] + 5*ell, dr)
    kern = kernel_func(r_conv, ell)
    v_bar_ext = np.interp(r_conv, r_kpc, v_bar_interp)
    
    conv_result = fftconvolve(v_bar_ext, kern, mode='same') * dr
    conv_interp = np.interp(r_kpc, r_conv, conv_result)
    
    v_dm = alpha * conv_interp + g_inf
    return np.maximum(v_dm, 0)

def v_total_general(r_kpc: np.ndarray, data: RotmodData, 
                   ell: float, alpha: float, g_inf: float, kernel_func) -> np.ndarray:
    """Total velocity with arbitrary kernel"""
    v_bar = v_baryonic(r_kpc, data)
    v_dm = v_dark_general(r_kpc, data, ell, alpha, g_inf, kernel_func)
    return np.sqrt(v_bar**2 + v_dm**2)

def fit_galaxy_with_kernel(data: RotmodData, kernel_func, kernel_name="unknown"):
    """Fit galaxy with specific kernel"""
    peak_v = np.max(data.v_obs)
    
    # Use same adaptive ranges as before
    ell_range = (0.5, 35.0)
    alpha_range = (0.01, 1.0)
    g_inf_range = (0.0, 0.6)
    
    def objective(params):
        ell, alpha, g_inf = params
        
        if not (ell_range[0] <= ell <= ell_range[1]): return 1e8
        if not (alpha_range[0] <= alpha <= alpha_range[1]): return 1e8  
        if not (g_inf_range[0] <= g_inf <= g_inf_range[1]): return 1e8
        
        try:
            v_pred = v_total_general(data.r_kpc, data, ell, alpha, g_inf, kernel_func)
            if np.any(~np.isfinite(v_pred)):
                return 1e8
            chi2 = np.sum(((data.v_obs - v_pred) / data.dv_obs)**2)
            return chi2
        except:
            return 1e8
    
    # Multiple starting points
    attempts = [
        (5.0, 0.15, 0.05),
        (2.0, 0.3, 0.1),
        (10.0, 0.1, 0.02),
        (15.0, 0.5, 0.0)
    ]
    
    best_params = None
    best_chi2 = 1e8
    
    for start in attempts:
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'fatol': 1e-8})
            if result.fun < best_chi2:
                best_chi2 = result.fun
                best_params = result.x
        except:
            continue
    
    return best_params, best_chi2

def detailed_kernel_analysis():
    """DEEPER INVESTIGATION - What's really going on?"""
    
    print("🔬 DETAILED KERNEL INVESTIGATION")
    print("=" * 50)
    
    # Kernel arsenal including unphysical one
    kernels = {
        'EXPONENTIAL': exponential_kernel,
        'GAUSSIAN': gaussian_kernel, 
        'POWER_LAW': power_law_kernel,
        'YUKAWA': yukawa_kernel,
        'LORENTZIAN': lorentzian_kernel,
        'STEP': step_kernel  # TOTALLY UNPHYSICAL
    }
    
    files = glob.glob("*_rotmod.dat")
    test_galaxies = files[:10]  # Smaller sample for detailed analysis
    
    print(f"Deep analysis of {len(test_galaxies)} galaxies...")
    
    # Test one galaxy in detail
    test_galaxy = test_galaxies[0]
    print(f"\nDetailed analysis of: {test_galaxy}")
    
    data = read_rotmod(test_galaxy)
    
    # Fit with each kernel and show the fitted parameters
    print(f"\n{'Kernel':<12} {'χ²':<8} {'ell':<8} {'α':<8} {'g∞':<8}")
    print("-" * 50)
    
    detailed_results = {}
    
    for name, kernel_func in kernels.items():
        params, chi2 = fit_galaxy_with_kernel(data, kernel_func, name)
        
        if params is not None:
            ell, alpha, g_inf = params
            print(f"{name:<12} {chi2:<8.1f} {ell:<8.2f} {alpha:<8.3f} {g_inf:<8.3f}")
            
            # Store results
            detailed_results[name] = {
                'params': params,
                'chi2': chi2,
                'v_pred': v_total_general(data.r_kpc, data, ell, alpha, g_inf, kernel_func)
            }
        else:
            print(f"{name:<12} FAILED")
            detailed_results[name] = None
    
    # Plot the actual fits
    plt.figure(figsize=(15, 12))
    
    # Kernel shapes comparison
    plt.subplot(2, 3, 1)
    r_test = np.linspace(0, 20, 200)
    for name, kernel_func in kernels.items():
        if name in detailed_results and detailed_results[name] is not None:
            ell = detailed_results[name]['params'][0]
            y = kernel_func(r_test, ell)
            plt.plot(r_test, y, label=f'{name} (ell={ell:.1f})', linewidth=2)
    
    plt.xlabel('r (kpc)')
    plt.ylabel('Kernel value')
    plt.title('Actual Fitted Kernels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 15)
    
    # Rotation curve fits
    plt.subplot(2, 3, (2, 3))
    plt.errorbar(data.r_kpc, data.v_obs, data.dv_obs, 
                fmt='ko', alpha=0.7, label='Observed', markersize=4)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (name, result) in enumerate(detailed_results.items()):
        if result is not None:
            plt.plot(data.r_kpc, result['v_pred'], 
                    color=colors[i % len(colors)], linewidth=2,
                    label=f"{name} (χ²={result['chi2']:.1f})")
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title(f'Rotation Curve Fits: {test_galaxy}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # χ² comparison
    plt.subplot(2, 3, 4)
    names = [name for name in detailed_results.keys() if detailed_results[name] is not None]
    chi2s = [detailed_results[name]['chi2'] for name in names]
    
    colors = ['red' if name == 'EXPONENTIAL' else 'gray' for name in names]
    bars = plt.bar(names, chi2s, color=colors, alpha=0.7)
    
    plt.ylabel('χ²')
    plt.title('Fit Quality Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Parameter correlations
    if len(names) > 3:
        ells = [detailed_results[name]['params'][0] for name in names]
        alphas = [detailed_results[name]['params'][1] for name in names]
        
        plt.subplot(2, 3, 5)
        for i, name in enumerate(names):
            color = 'red' if name == 'EXPONENTIAL' else 'blue'
            size = 100 if name == 'EXPONENTIAL' else 50
            plt.scatter(ells[i], alphas[i], c=color, s=size, 
                       label=name, alpha=0.7)
        
        plt.xlabel('ell parameter')
        plt.ylabel('α parameter')
        plt.title('Parameter Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Residuals for best fits
    plt.subplot(2, 3, 6)
    if detailed_results['EXPONENTIAL'] is not None:
        exp_residuals = data.v_obs - detailed_results['EXPONENTIAL']['v_pred']
        plt.scatter(data.r_kpc, exp_residuals, alpha=0.7, c='red', 
                   label='EXPONENTIAL')
    
    # Compare to another kernel
    other_kernels = [name for name in names if name != 'EXPONENTIAL']
    if other_kernels and detailed_results[other_kernels[0]] is not None:
        other_name = other_kernels[0]
        other_residuals = data.v_obs - detailed_results[other_name]['v_pred']
        plt.scatter(data.r_kpc, other_residuals, alpha=0.7, c='blue',
                   label=other_name)
    
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Residuals (km/s)')
    plt.title('Fit Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_kernel_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # CRITICAL ANALYSIS
    print(f"\n🚨 CRITICAL FINDINGS:")
    
    if detailed_results['STEP'] is not None:
        print(f"⚠️  WARNING: Even unphysical STEP kernel fits well!")
        print(f"   This suggests your model might be overfitting.")
    
    # Check parameter variation
    valid_results = {k: v for k, v in detailed_results.items() if v is not None}
    
    if len(valid_results) > 2:
        ell_range = [v['params'][0] for v in valid_results.values()]
        alpha_range = [v['params'][1] for v in valid_results.values()]
        
        ell_variation = (max(ell_range) - min(ell_range)) / np.mean(ell_range)
        alpha_variation = (max(alpha_range) - min(alpha_range)) / np.mean(alpha_range)
        
        print(f"\n📊 PARAMETER VARIATIONS:")
        print(f"   ell variation: {ell_variation:.2%}")
        print(f"   α variation: {alpha_variation:.2%}")
        
        if ell_variation > 0.5 or alpha_variation > 0.5:
            print(f"⚠️  Large parameter variations suggest kernel choice matters!")
        else:
            print(f"⚠️  Small parameter variations suggest overfitting!")
    
    return detailed_results

if __name__ == "__main__":
    results = detailed_kernel_analysis()