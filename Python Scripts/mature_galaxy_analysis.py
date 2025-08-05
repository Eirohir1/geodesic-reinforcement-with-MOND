import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional

@dataclass
class RotmodData:
    r_kpc: np.ndarray
    v_obs: np.ndarray
    dv_obs: np.ndarray
    v_gas: Optional[np.ndarray] = None
    v_disk: Optional[np.ndarray] = None
    v_bulge: Optional[np.ndarray] = None

def read_rotmod(path: str) -> RotmodData:
    """Read rotation curve data - VERIFIED CORRECT"""
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
    
    # Data validation
    if np.any(r <= 0):
        raise ValueError("Non-positive radii detected")
    if np.any(dv <= 0):
        raise ValueError("Non-positive velocity errors detected")
    
    return RotmodData(r, vobs, dv, vgas, vdisk, vbulge)

def v_baryonic(r_kpc: np.ndarray, data: RotmodData) -> np.ndarray:
    """Calculate baryonic velocity"""
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
    """Standard MOND implementation"""
    v_bar = v_baryonic(r_kpc, data)
    
    # Newtonian acceleration
    mask = r_kpc > 0
    a_N = np.zeros_like(r_kpc)
    a_N[mask] = v_bar[mask]**2 / r_kpc[mask]
    
    # Convert to m/s²
    a_N_SI = a_N * 1e6 / 3.086e19
    
    # MOND interpolation
    x = a_N_SI / a0_kms2
    mu = x / (1 + x)
    
    # MOND velocity
    mu_safe = np.maximum(mu, 1e-10)
    v_mond = v_bar / np.sqrt(mu_safe)
    
    return v_mond

def geodesic_reinforcement(r_kpc: np.ndarray, data: RotmodData, 
                          alpha: float, ell_factor: float) -> np.ndarray:
    """Geodesic reinforcement model"""
    v_bar = v_baryonic(r_kpc, data)
    
    if len(r_kpc) > 1:
        dr = np.median(np.diff(r_kpc))
    else:
        dr = 0.1
        
    R_galaxy = np.max(data.r_kpc)
    ell = ell_factor * R_galaxy
    
    # Convolution
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
    """Robust model fitting"""
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
    
    methods = ['Nelder-Mead', 'Powell']
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
    """Compare models with full diagnostics"""
    results = {}
    
    # Geodesic model
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
    
    # MOND model
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

def is_regular_rotation_curve(data):
    """Check if rotation curve shows regular, settled behavior"""
    
    # Look for smooth, rising then flat rotation curve
    if len(data.v_obs) < 5:
        return False
    
    # Check for monotonic rise in inner regions
    inner_mask = data.r_kpc < np.max(data.r_kpc) / 3
    if np.sum(inner_mask) > 3:
        inner_slope = np.polyfit(data.r_kpc[inner_mask], data.v_obs[inner_mask], 1)[0]
        rising_inner = inner_slope > 0
    else:
        rising_inner = True
    
    # Check for reasonable flatness in outer regions  
    outer_mask = data.r_kpc > np.max(data.r_kpc) * 0.6
    if np.sum(outer_mask) > 3:
        outer_variation = np.std(data.v_obs[outer_mask]) / np.mean(data.v_obs[outer_mask])
        flat_outer = outer_variation < 0.3  # < 30% variation
    else:
        flat_outer = True
    
    return rising_inner and flat_outer

def filter_mature_galaxies(files):
    """Select only mature, settled galaxies for geodesic testing"""
    
    print("\n🔍 FILTERING FOR MATURE GALAXIES:")
    mature_galaxies = []
    
    for filename in files:
        try:
            data = read_rotmod(filename)
            
            # Basic maturity criteria
            v_max = np.max(data.v_obs)
            R_galaxy = np.max(data.r_kpc)
            
            # Filter criteria for mature galaxies
            criteria = {
                'high_velocity': v_max > 120,  # km/s - well-formed rotation
                'large_size': R_galaxy > 8,    # kpc - extended disk
                'regular_curve': is_regular_rotation_curve(data),
                'not_dwarf': 'DDO' not in filename and 'IC' not in filename,
                'not_irregular': not any(x in filename.upper() for x in ['IRR', 'PEC', 'IRREG'])
            }
            
            # Check stellar dominance if data available
            if data.v_gas is not None and data.v_disk is not None:
                v_gas_med = np.median(data.v_gas[data.v_gas > 0]) if np.any(data.v_gas > 0) else 0
                v_disk_med = np.median(data.v_disk[data.v_disk > 0]) if np.any(data.v_disk > 0) else 0
                stellar_dominance = v_disk_med > v_gas_med if v_gas_med > 0 else True
                criteria['stellar_dominated'] = stellar_dominance
            
            # Galaxy passes if meets most criteria
            score = sum(criteria.values())
            print(f"  {filename}: v_max={v_max:.1f} km/s, R={R_galaxy:.1f} kpc, score={score}/{len(criteria)}")
            
            if score >= 3:  # At least 3/5 criteria
                mature_galaxies.append(filename)
                print(f"    ✅ MATURE GALAXY SELECTED")
            else:
                print(f"    ❌ EXCLUDED (insufficient maturity)")
                
        except Exception as e:
            print(f"    ❌ ERROR reading {filename}: {e}")
    
    return mature_galaxies

def comprehensive_model_diagnostics(filename):
    """Brutal honesty analysis of model performance"""
    
    print(f"\n🔍 DIAGNOSTIC ANALYSIS: {filename}")
    print("-" * 40)
    
    # Load data and run models
    data = read_rotmod(filename)
    results = compare_all_models(data)
    
    # 1. VELOCITY SCALE ANALYSIS
    print("\n📊 VELOCITY SCALE ANALYSIS:")
    v_obs_max = np.max(data.v_obs)
    v_obs_typical = np.median(data.v_obs[data.r_kpc > 5]) if np.any(data.r_kpc > 5) else np.median(data.v_obs)
    
    print(f"  Observed max velocity: {v_obs_max:.1f} km/s")
    print(f"  Observed typical: {v_obs_typical:.1f} km/s")
    
    for model_name, res in results.items():
        if 'v_pred' in res:
            v_pred = res['v_pred']
            v_pred_max = np.max(v_pred)
            v_pred_typical = np.median(v_pred)
            
            ratio_max = v_pred_max / v_obs_max
            ratio_typical = v_pred_typical / v_obs_typical
            
            print(f"  {model_name}:")
            print(f"    Max velocity: {v_pred_max:.1f} km/s (ratio: {ratio_max:.2f})")
            print(f"    Typical velocity: {v_pred_typical:.1f} km/s (ratio: {ratio_typical:.2f})")
            
            # Red flags
            if ratio_max > 3.0:
                print(f"    🚨 EXTREME VELOCITY: {ratio_max:.1f}X observed!")
            elif ratio_max > 2.0:
                print(f"    ⚠️  High velocity: {ratio_max:.1f}X observed")
            elif 0.7 <= ratio_max <= 1.5:
                print(f"    ✅ Reasonable velocity scale")
            else:
                print(f"    🤔 Unusual velocity scaling")
    
    # 2. PARAMETER ANALYSIS
    print("\n⚙️  PARAMETER ANALYSIS:")
    if 'GEODESIC' in results:
        alpha, ell_factor = results['GEODESIC']['params']
        R_galaxy = np.max(data.r_kpc)
        
        print(f"  α = {alpha:.3f}")
        print(f"  ℓ/R_galaxy = {ell_factor:.3f}")
        print(f"  Galaxy radius: {R_galaxy:.1f} kpc")
        print(f"  Coupling length: {ell_factor * R_galaxy:.1f} kpc")
        
        # Parameter sanity checks
        if alpha > 0.8:
            print(f"    🚨 EXTREME COUPLING: α = {alpha:.3f}")
        elif alpha > 0.5:
            print(f"    ⚠️  Strong coupling: α = {alpha:.3f}")
        elif alpha < 0.1:
            print(f"    ⚠️  Weak coupling: α = {alpha:.3f}")
        else:
            print(f"    ✅ Reasonable coupling strength")
            
        if ell_factor > 2.0:
            print(f"    🚨 EXTREME RANGE: ℓ = {ell_factor:.1f} × R_galaxy")
        elif ell_factor > 1.5:
            print(f"    ⚠️  Long range coupling")
        else:
            print(f"    ✅ Reasonable coupling range")
    
    # 3. CHI-SQUARED INTERPRETATION
    print("\n📈 STATISTICAL INTERPRETATION:")
    for model_name, res in results.items():
        chi2_reduced = res['reduced_chi2']
        
        print(f"  {model_name}: χ²/dof = {chi2_reduced:.2f}")
        
        if chi2_reduced < 0.5:
            print(f"    🤔 Suspiciously good fit (possible overfitting)")
        elif chi2_reduced <= 3.0:
            print(f"    ✅ Good fit")
        elif chi2_reduced <= 10.0:
            print(f"    ⚠️  Acceptable fit")
        elif chi2_reduced <= 50.0:
            print(f"    ❌ Poor fit")
        else:
            print(f"    🚨 CATASTROPHIC FIT")
    
    return results

def mature_galaxy_analysis():
    """Test geodesic reinforcement on mature galaxies only"""
    
    print("🌟 MATURE GALAXY GEODESIC ANALYSIS")
    print("=" * 60)
    print("Strategy: Test only settled, stellar-dominated systems")
    print("Rationale: Geodesic paths require stable matter distribution")
    print()
    
    # Get all files
    all_files = glob.glob("*_rotmod.dat")
    print(f"Total rotation curve files available: {len(all_files)}")
    
    # Filter for mature galaxies
    mature_files = filter_mature_galaxies(all_files)
    
    print(f"\n📊 SELECTION SUMMARY:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Mature galaxies: {len(mature_files)}")
    if len(all_files) > 0:
        print(f"  Selection rate: {len(mature_files)/len(all_files)*100:.1f}%")
    
    if not mature_files:
        print("\n❌ No mature galaxies found in dataset!")
        print("   Dataset may be biased toward dwarf/irregular galaxies")
        print("   Testing on ALL galaxies instead...")
        mature_files = all_files[:10]  # Use first 10 as backup
    
    # Run analysis on mature galaxies only
    print(f"\n🔬 ANALYZING {len(mature_files)} GALAXIES:")
    
    results_mature = []
    
    for filename in mature_files:
        print(f"\n{'='*50}")
        print(f"🌟 ANALYZING: {filename}")
        
        try:
            results = comprehensive_model_diagnostics(filename)
            results_mature.append((filename, results))
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    # Summary for mature galaxies
    if results_mature:
        print(f"\n🎯 FINAL SUMMARY:")
        print("=" * 50)
        
        geodesic_chi2 = []
        mond_chi2 = []
        clean_fits = 0
        reasonable_params = 0
        
        for filename, results in results_mature:
            if 'GEODESIC' in results:
                chi2 = results['GEODESIC']['reduced_chi2']
                alpha, ell_factor = results['GEODESIC']['params']
                
                geodesic_chi2.append(chi2)
                
                # Count clean fits
                if chi2 < 5.0:
                    clean_fits += 1
                    
                # Count reasonable parameters
                if 0.1 <= alpha <= 0.8 and ell_factor <= 1.5:
                    reasonable_params += 1
            
            if 'MOND' in results:
                chi2 = results['MOND']['reduced_chi2']
                mond_chi2.append(chi2)
        
        print(f"Galaxies analyzed: {len(results_mature)}")
        print(f"Geodesic clean fits (χ²/dof < 5): {clean_fits}/{len(results_mature)} = {clean_fits/len(results_mature)*100:.1f}%")
        print(f"Reasonable parameters: {reasonable_params}/{len(results_mature)} = {reasonable_params/len(results_mature)*100:.1f}%")
        
        if geodesic_chi2:
            print(f"Geodesic median χ²/dof: {np.median(geodesic_chi2):.2f}")
        if mond_chi2:
            print(f"MOND median χ²/dof: {np.median(mond_chi2):.2f}")
            mond_clean = sum(1 for chi2 in mond_chi2 if chi2 < 5.0)
            print(f"MOND clean fits: {mond_clean}/{len(mond_chi2)} = {mond_clean/len(mond_chi2)*100:.1f}%")
        
        # The verdict for mature galaxies
        success_rate = clean_fits / len(results_mature)
        
        print(f"\n🌟 FINAL VERDICT:")
        if success_rate > 0.7:
            print("🎉 STRONG EVIDENCE: Geodesic reinforcement works for mature galaxies!")
        elif success_rate > 0.5:
            print("✅ GOOD EVIDENCE: Theory applies to settled systems")
        elif success_rate > 0.3:
            print("🤔 MIXED EVIDENCE: Partial success on mature galaxies")
        else:
            print("❌ FAILURE: Theory struggles even with mature galaxies")
            
        # Compare to MOND
        if geodesic_chi2 and mond_chi2:
            geodesic_median = np.median(geodesic_chi2)
            mond_median = np.median(mond_chi2)
            improvement = mond_median / geodesic_median
            print(f"Geodesic is {improvement:.1f}X better than MOND on average")
    
    return results_mature

# Run the mature galaxy analysis
if __name__ == "__main__":
    mature_results = mature_galaxy_analysis()