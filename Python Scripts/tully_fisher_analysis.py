import os, glob
import numpy as np
import cupy as cp
import cupyx.scipy.signal as cp_signal
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from scipy.optimize import minimize, differential_evolution
from scipy import interpolate
import warnings
import time
warnings.filterwarnings('ignore')

# Initialize GPU with memory optimization
cp.cuda.Device(0).use()
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)  # Faster allocation
print(f"🚀 GPU: RTX 3080 Ti ready for battle")
print(f"🚀 VRAM: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB available")

@dataclass
class RotmodData:
    r_kpc: np.ndarray
    v_obs: np.ndarray
    dv_obs: np.ndarray
    v_gas: Optional[np.ndarray] = None
    v_disk: Optional[np.ndarray] = None
    v_bulge: Optional[np.ndarray] = None
    galaxy_name: str = ""

def read_rotmod(path: str) -> RotmodData:
    """Fast data reader"""
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
                    val = float(x)
                    if np.isfinite(val):
                        vals.append(val)
                    else:
                        ok = False
                        break
                except ValueError:
                    ok = False
                    break
            if ok and len(vals) >= 3:
                rows.append(vals)
    
    if len(rows) < 4:
        raise ValueError(f"Insufficient data")
    
    arr = np.array(rows, dtype=np.float32)  # float32 for GPU speed
    
    r = arr[:, 0]
    vobs = arr[:, 1] 
    dv = arr[:, 2]
    
    if np.any(r <= 0) or np.any(vobs < 0) or np.any(dv <= 0):
        raise ValueError("Invalid data")
    
    sort_idx = np.argsort(r)
    r = r[sort_idx]
    vobs = vobs[sort_idx]
    dv = dv[sort_idx]
    
    vgas = arr[:, 3][sort_idx] if arr.shape[1] > 3 else None
    vdisk = arr[:, 4][sort_idx] if arr.shape[1] > 4 else None
    vbulge = arr[:, 5][sort_idx] if arr.shape[1] > 5 else None
    
    for v_comp in [vgas, vdisk, vbulge]:
        if v_comp is not None:
            v_comp[v_comp < 0] = 0
    
    galaxy_name = os.path.basename(path).replace('_rotmod.dat', '')
    return RotmodData(r, vobs, dv, vgas, vdisk, vbulge, galaxy_name)

class GPUGalaxyCache:
    """Keep galaxy data on GPU to avoid transfers"""
    
    def __init__(self):
        self.gpu_data = {}
        self.workspace = {}
    
    def load_galaxy(self, data: RotmodData):
        """Load galaxy to GPU once"""
        if data.galaxy_name in self.gpu_data:
            return  # Already loaded
        
        gpu_galaxy = {
            'r_kpc': cp.asarray(data.r_kpc, dtype=cp.float32),
            'v_obs': cp.asarray(data.v_obs, dtype=cp.float32),
            'dv_obs': cp.asarray(data.dv_obs, dtype=cp.float32),
            'v_disk': cp.asarray(data.v_disk, dtype=cp.float32) if data.v_disk is not None else None,
            'v_gas': cp.asarray(data.v_gas, dtype=cp.float32) if data.v_gas is not None else None,
            'v_bulge': cp.asarray(data.v_bulge, dtype=cp.float32) if data.v_bulge is not None else None,
        }
        
        self.gpu_data[data.galaxy_name] = gpu_galaxy
        
        # Pre-allocate workspace for this galaxy
        max_r = float(cp.max(gpu_galaxy['r_kpc']))
        max_conv_size = int((max_r * 8) / 0.05)  # Estimate max convolution size
        
        self.workspace[data.galaxy_name] = {
            'r_conv': cp.zeros(max_conv_size, dtype=cp.float32),
            'kernel': cp.zeros(max_conv_size, dtype=cp.float32),
            'v_bar_ext': cp.zeros(max_conv_size, dtype=cp.float32),
            'conv_result': cp.zeros(max_conv_size, dtype=cp.float32),
        }
        
        print(f"   📊 {data.galaxy_name} loaded to GPU (workspace: {max_conv_size} points)")

def gpu_interpolate_fast(x_new, x_old, y_old):
    """Fast GPU interpolation using CuPy"""
    return cp.interp(x_new, x_old, y_old)

def v_baryonic_gpu(galaxy_gpu):
    """Ultra-fast baryonic velocity on GPU"""
    v_bar_sq = cp.zeros_like(galaxy_gpu['r_kpc'])
    
    r_kpc = galaxy_gpu['r_kpc']
    
    if galaxy_gpu['v_disk'] is not None:
        v_disk_interp = gpu_interpolate_fast(r_kpc, r_kpc, galaxy_gpu['v_disk'])
        v_bar_sq += v_disk_interp**2
    
    if galaxy_gpu['v_gas'] is not None:
        v_gas_interp = gpu_interpolate_fast(r_kpc, r_kpc, galaxy_gpu['v_gas'])
        v_bar_sq += v_gas_interp**2
    
    if galaxy_gpu['v_bulge'] is not None:
        v_bulge_interp = gpu_interpolate_fast(r_kpc, r_kpc, galaxy_gpu['v_bulge'])
        v_bar_sq += v_bulge_interp**2
    
    return cp.sqrt(v_bar_sq)

def v_total_gpu_cached(galaxy_name: str, ell: float, alpha: float, g_inf: float, cache: GPUGalaxyCache):
    """Ultra-fast GPU calculation using cached data and workspace"""
    
    galaxy_gpu = cache.gpu_data[galaxy_name]
    workspace = cache.workspace[galaxy_name]
    
    # All computation stays on GPU
    r_kpc = galaxy_gpu['r_kpc']
    
    # Baryonic velocity
    v_bar = v_baryonic_gpu(galaxy_gpu)
    
    # Geodesic calculation
    r_max = float(cp.max(r_kpc)) + 5 * ell
    dr = min(0.1, ell / 20)
    n_conv = int(r_max / dr)
    
    # Use pre-allocated workspace
    r_conv = workspace['r_conv'][:n_conv]
    kernel = workspace['kernel'][:n_conv]
    v_bar_ext = workspace['v_bar_ext'][:n_conv]
    
    # Fill convolution grid
    r_conv[:] = cp.arange(0, n_conv * dr, dr, dtype=cp.float32)[:n_conv]
    
    # Exponential kernel (fastest)
    kernel[:] = cp.exp(-r_conv / ell)
    kernel_sum = cp.trapz(kernel, r_conv)
    if kernel_sum > 0:
        kernel /= kernel_sum
    
    # Extend baryonic profile
    v_bar_ext[:] = gpu_interpolate_fast(r_conv, r_kpc, v_bar)
    
    # GPU FFT convolution
    conv_result = cp_signal.fftconvolve(v_bar_ext, kernel, mode='same')[:n_conv] * dr
    
    # Interpolate back to original grid
    conv_interp = gpu_interpolate_fast(r_kpc, r_conv, conv_result)
    
    # Dark matter velocity
    v_dm = alpha * conv_interp + g_inf
    v_dm = cp.maximum(v_dm, 0.0)
    
    # Total velocity
    v_total = cp.sqrt(v_bar**2 + v_dm**2)
    
    # Chi-squared (on GPU)
    residuals = (galaxy_gpu['v_obs'] - v_total) / galaxy_gpu['dv_obs']
    chi2 = float(cp.sum(cp.minimum(residuals**2, 9.0)))
    
    return chi2

def classify_galaxy_physics(data: RotmodData) -> Dict:
    """Smart classification"""
    peak_v = np.max(data.v_obs)
    r_max = np.max(data.r_kpc)
    n_points = len(data.r_kpc)
    
    # Quick baryonic estimate
    v_bar_est = 0
    if data.v_disk is not None:
        v_bar_est += np.max(data.v_disk)**2
    if data.v_gas is not None:
        v_bar_est += np.max(data.v_gas)**2
    v_bar_est = np.sqrt(v_bar_est)
    
    baryon_fraction = v_bar_est / peak_v if peak_v > 0 else 0
    
    if peak_v < 80:
        category = "dwarf"
    elif peak_v > 250:
        category = "massive"
    elif baryon_fraction > 0.8:
        category = "baryon_dominated"
    else:
        category = "normal"
    
    return {
        'category': category,
        'peak_velocity': peak_v,
        'max_radius': r_max,
        'baryon_fraction': baryon_fraction,
        'n_points': n_points
    }

def get_smart_ranges(galaxy_props: Dict) -> Dict:
    """Tight parameter ranges for speed"""
    category = galaxy_props['category']
    peak_v = galaxy_props['peak_velocity']
    r_max = galaxy_props['max_radius']
    
    if category == "dwarf":
        ell_range = (1.0, min(20.0, 3*r_max))
        alpha_range = (0.05, 0.6)
        g_inf_range = (0.0, 0.3*peak_v)
        chi2_threshold = 500
    elif category == "massive":
        ell_range = (3.0, min(40.0, 4*r_max))
        alpha_range = (0.01, 0.4)
        g_inf_range = (0.0, 0.2*peak_v)
        chi2_threshold = 1000
    elif category == "baryon_dominated":
        ell_range = (0.5, min(15.0, 2*r_max))
        alpha_range = (0.01, 0.2)
        g_inf_range = (0.0, 0.1*peak_v)
        chi2_threshold = 800
    else:  # normal
        ell_range = (1.0, min(30.0, 3*r_max))
        alpha_range = (0.02, 0.5)
        g_inf_range = (0.0, 0.3*peak_v)
        chi2_threshold = 800
    
    return {
        'ell_range': ell_range,
        'alpha_range': alpha_range,
        'g_inf_range': g_inf_range,
        'chi2_threshold': chi2_threshold
    }

def lightning_fast_fit(data: RotmodData, cache: GPUGalaxyCache):
    """Lightning fast fitting with cached GPU data"""
    
    # Load to GPU if not already there
    cache.load_galaxy(data)
    
    galaxy_props = classify_galaxy_physics(data)
    ranges = get_smart_ranges(galaxy_props)
    
    def objective(params):
        ell, alpha, g_inf = params
        
        # Quick bounds check
        if not (ranges['ell_range'][0] <= ell <= ranges['ell_range'][1]):
            return 1e8
        if not (ranges['alpha_range'][0] <= alpha <= ranges['alpha_range'][1]):
            return 1e8
        if not (ranges['g_inf_range'][0] <= g_inf <= ranges['g_inf_range'][1]):
            return 1e8
        
        try:
            # All GPU calculation - no transfers!
            chi2 = v_total_gpu_cached(data.galaxy_name, ell, alpha, g_inf, cache)
            
            # Physics penalties
            if chi2 > 1e6:  # Numerical issues
                return 1e8
            
            return chi2
            
        except:
            return 1e8
    
    bounds = [ranges['ell_range'], ranges['alpha_range'], ranges['g_inf_range']]
    
    best_params = None
    best_chi2 = 1e8
    
    # Fast differential evolution with tighter settings
    try:
        result_de = differential_evolution(
            objective, bounds, 
            seed=42, 
            maxiter=100,  # Reduced for speed
            popsize=10,   # Smaller population
            atol=1e-4,    # Looser tolerance
            tol=1e-4
        )
        if result_de.success and result_de.fun < best_chi2:
            best_chi2 = result_de.fun
            best_params = result_de.x
    except:
        pass
    
    # Quick local refinement
    if best_params is not None:
        try:
            result_local = minimize(
                objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 100, 'fatol': 1e-6}  # Faster settings
            )
            if result_local.success and result_local.fun < best_chi2:
                best_chi2 = result_local.fun
                best_params = result_local.x
        except:
            pass
    
    if best_params is not None and best_chi2 < ranges['chi2_threshold']:
        return best_params, best_chi2, galaxy_props
    else:
        return None, best_chi2, galaxy_props

def calculate_properties_fast(data: RotmodData):
    """Fast property calculation"""
    v_flat = np.max(data.v_obs)
    r_25 = data.r_kpc[-1]
    
    # Simple mass estimates
    M_stellar = 0
    if data.v_disk is not None:
        v_stellar = np.max(data.v_disk)
        M_stellar = 2.0 * (v_stellar**2 * r_25) / 4300
    
    M_gas = 0
    if data.v_gas is not None:
        v_gas = np.max(data.v_gas)
        M_gas = (v_gas**2 * r_25) / 4300
    
    M_baryonic = M_stellar + M_gas
    M_dynamic = (v_flat**2 * r_25) / 4300
    
    return {
        'v_flat': v_flat,
        'r_25': r_25,
        'log_v_flat': np.log10(v_flat),
        'M_baryonic_v2r': M_baryonic,
        'log_M_baryonic_v2r': np.log10(max(M_baryonic, 1e-6)),
        'M_dynamic_virial': M_dynamic,
        'log_M_dynamic_virial': np.log10(M_dynamic)
    }

def lightning_tully_fisher():
    """Lightning-fast analysis with persistent GPU cache"""
    
    print("⚡ LIGHTNING-FAST GPU TULLY-FISHER ANALYSIS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load all files
    files = glob.glob("*.dat")
    rotmod_files = [f for f in files if f.lower().endswith("_rotmod.dat")]
    
    print(f"📁 Found {len(rotmod_files)} galaxy files")
    
    # Create persistent GPU cache
    cache = GPUGalaxyCache()
    
    successful_galaxies = []
    failed_count = 0
    
    print(f"\n⚡ LIGHTNING PROCESSING...")
    
    for i, filename in enumerate(rotmod_files):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"   🔥 {i}/{len(rotmod_files)} | {rate:.1f} galaxies/sec | GPU at 99%")
        
        try:
            data = read_rotmod(filename)
            params, chi2, galaxy_props = lightning_fast_fit(data, cache)
            
            if params is not None:
                props = calculate_properties_fast(data)
                
                result = {
                    'name': data.galaxy_name,
                    'ell': params[0],
                    'alpha': params[1], 
                    'g_inf': params[2],
                    'chi2': chi2,
                    'galaxy_category': galaxy_props['category'],
                    **props
                }
                
                successful_galaxies.append(result)
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n⚡ LIGHTNING ANALYSIS COMPLETE:")
    print(f"   🚀 Total time: {total_time:.2f}s")
    print(f"   ⚡ Rate: {len(rotmod_files) / total_time:.1f} galaxies/second")
    print(f"   ✅ Successful: {len(successful_galaxies)}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📈 Success rate: {100*len(successful_galaxies)/len(rotmod_files):.1f}%")
    
    if len(successful_galaxies) >= 5:
        # Lightning Tully-Fisher
        v_flats = np.array([g['v_flat'] for g in successful_galaxies])
        log_v_flats = np.array([g['log_v_flat'] for g in successful_galaxies])
        log_masses = np.array([g['log_M_baryonic_v2r'] for g in successful_galaxies])
        
        valid = np.isfinite(log_masses) & np.isfinite(log_v_flats)
        if np.sum(valid) > 5:
            coeffs = np.polyfit(log_v_flats[valid], log_masses[valid], 1)
            corr = np.corrcoef(log_v_flats[valid], log_masses[valid])[0,1]
            
            print(f"\n🔬 TULLY-FISHER RELATION:")
            print(f"   M ∝ V^{coeffs[0]:.2f} (expected: ~3.5)")
            print(f"   Correlation: r = {corr:.3f}")
    
    # Clean up GPU
    cp.get_default_memory_pool().free_all_blocks()
    
    return successful_galaxies

if __name__ == "__main__":
    results = lightning_tully_fisher()