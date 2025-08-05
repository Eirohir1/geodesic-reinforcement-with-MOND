import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from numba import cuda, float32
import math

# Set CuPy memory pool (8GB = 8 * 1024**3 bytes)
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=8 * 1024**3)  # 8GB limit

# GPU kernel for force calculation
@cuda.jit
def calculate_forces_kernel(positions, masses, accelerations, G, M_bh, N, softening):
    """
    CUDA kernel to calculate gravitational forces
    Each thread calculates acceleration for one particle
    """
    
    i = cuda.grid(1)
    
    if i < N:
        ax = 0.0
        ay = 0.0
        
        # Position of particle i
        xi = positions[i, 0]
        yi = positions[i, 1]
        
        # Central black hole force
        r_bh = math.sqrt(xi*xi + yi*yi)
        if r_bh > softening:
            a_bh = -G * M_bh / (r_bh * r_bh * r_bh)
            ax += a_bh * xi
            ay += a_bh * yi
        
        # Forces from all other stars
        for j in range(N):
            if i != j:
                dx = xi - positions[j, 0]
                dy = yi - positions[j, 1]
                
                r2 = dx*dx + dy*dy + softening*softening
                r = math.sqrt(r2)
                r3 = r2 * r
                
                a = -G * masses[j] / r3
                ax += a * dx
                ay += a * dy
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay


class GPUGalaxySimulation:
    """
    GPU-accelerated N-body galaxy simulation
    Optimized for RTX 3080 Ti with 12GB VRAM
    """
    
    def __init__(self, 
                 M_bh=4e6,
                 M_total=6e10,
                 R_galaxy=15.0,
                 N_stars=2000,
                 dt=0.01,
                 use_gpu=True):
        
        self.G = 4.3e-6  # Gravitational constant in galaxy units
        self.M_bh = M_bh
        self.M_total = M_total
        self.R_galaxy = R_galaxy
        self.N_stars = N_stars
        self.dt = dt
        self.use_gpu = use_gpu
        self.R_scale = R_galaxy / 3.0
        
        # Choose compute backend
        if use_gpu:
            self.xp = cp  # CuPy for GPU
            print(f"🚀 Using GPU acceleration")
            print(f"   VRAM: {cp.cuda.Device().mem_info[1] // 1024**3} GB")
        else:
            self.xp = np  # NumPy for CPU
            print("🖥️  Using CPU computation")
        
        self.initialize_galaxy()
    
    def initialize_galaxy(self):
        """Initialize galaxy structure on GPU"""
        
        print(f"Initializing {self.N_stars} particles...")
        
        # Generate on CPU first (easier random generation)
        radii = np.random.exponential(self.R_scale, self.N_stars)
        radii = np.clip(radii, 0.1, self.R_galaxy)
        
        angles = np.random.uniform(0, 2*np.pi, self.N_stars)
        
        # 2D positions
        positions_cpu = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ]).astype(np.float32)
        
        # Stellar masses (Salpeter IMF)
        masses_cpu = self.generate_stellar_masses().astype(np.float32)
        
        # Calculate initial velocities on CPU (complex potential calculation)
        velocities_cpu = self.calculate_initial_velocities_cpu(positions_cpu, masses_cpu)
        
        # Transfer to GPU
        self.positions = self.xp.array(positions_cpu)
        self.velocities = self.xp.array(velocities_cpu)
        self.masses = self.xp.array(masses_cpu)
        
        # Initialize acceleration arrays
        self.accelerations = self.xp.zeros((self.N_stars, 2), dtype=self.xp.float32)
        
        self.time = 0.0
        
        print(f"Galaxy initialized on {'GPU' if self.use_gpu else 'CPU'}")
        print(f"Mass range: {float(self.xp.min(self.masses)):.1e} - {float(self.xp.max(self.masses)):.1e} M☉")
        print(f"Velocity range: {float(self.xp.min(self.xp.linalg.norm(self.velocities, axis=1))):.1f} - {float(self.xp.max(self.xp.linalg.norm(self.velocities, axis=1))):.1f} km/s")
    
    def generate_stellar_masses(self):
        """Generate realistic stellar mass function"""
        # Salpeter IMF: dN/dM ∝ M^(-2.35)
        alpha = 2.35
        m_min, m_max = 0.1, 20.0
        
        u = np.random.random(self.N_stars)
        masses = m_min * (1 + u * ((m_max/m_min)**(1-alpha) - 1))**(1/(1-alpha))
        
        # Normalize to total mass
        masses = masses * (self.M_total / np.sum(masses))
        return masses
    
    def calculate_initial_velocities_cpu(self, positions, masses):
        """Calculate stable initial velocities using combined potential"""
        
        velocities = np.zeros_like(positions, dtype=np.float32)
        
        for i, pos in enumerate(positions):
            r = np.linalg.norm(pos)
            
            if r > 0.1:
                # Combined BH + stellar disk potential
                v_bh = np.sqrt(self.G * self.M_bh / r)
                
                # Enclosed stellar mass (exponential disk approximation)
                M_stellar_enclosed = self.M_total * (1 - np.exp(-r/self.R_scale) * (1 + r/self.R_scale))
                v_stellar = np.sqrt(self.G * M_stellar_enclosed / r)
                
                # Combined circular velocity
                v_circular = np.sqrt(v_bh**2 + v_stellar**2)
                
                # Tangential direction
                if r > 0:
                    tangent = np.array([-pos[1], pos[0]], dtype=np.float32) / r
                    velocities[i] = v_circular * tangent
                    
                    # Small random perturbation
                    perturbation = np.random.normal(0, v_circular * 0.03, 2).astype(np.float32)
                    velocities[i] += perturbation
        
        return velocities
    
    def calculate_forces_gpu(self):
        """Calculate gravitational forces using GPU kernel"""
        
        if self.use_gpu:
            # CUDA kernel execution
            threads_per_block = 256
            blocks_per_grid = (self.N_stars + threads_per_block - 1) // threads_per_block
            
            softening = 0.01
            calculate_forces_kernel[blocks_per_grid, threads_per_block](
                self.positions, self.masses, self.accelerations, 
                self.G, self.M_bh, self.N_stars, softening
            )
            
            cp.cuda.Device().synchronize()
        else:
            # CPU fallback using vectorized operations
            self.calculate_forces_cpu()
    
    def calculate_forces_cpu(self):
        """CPU fallback for force calculation"""
        
        self.accelerations.fill(0)
        
        # Central BH forces
        r_vectors = self.positions  # Shape: (N, 2)
        r_magnitudes = self.xp.linalg.norm(r_vectors, axis=1, keepdims=True)
        r_magnitudes = self.xp.maximum(r_magnitudes, 0.01)  # Softening
        
        a_bh = -self.G * self.M_bh / (r_magnitudes**3)
        self.accelerations += a_bh * r_vectors
        
        # Star-star forces (vectorized but still O(N²))
        for i in range(self.N_stars):
            r_ij = self.positions[i:i+1] - self.positions  # Broadcasting
            r_mag = self.xp.linalg.norm(r_ij, axis=1, keepdims=True)
            r_mag = self.xp.maximum(r_mag, 0.01)  # Softening
            
            # Exclude self-interaction
            mask = self.xp.arange(self.N_stars) != i
            
            a_ij = -self.G * self.masses[mask] / (r_mag[mask]**3)
            self.accelerations[i] += self.xp.sum(a_ij * r_ij[mask], axis=0)
    
    def step_simulation(self, method='discrete'):
        """Advance simulation by one time step"""
        
        if method == 'discrete':
            # Use individual stellar masses
            self.calculate_forces_gpu()
        else:
            # Use smooth mass distribution
            self.calculate_smooth_forces()
        
        # Leapfrog integration
        self.velocities += self.accelerations * self.dt
        self.positions += self.velocities * self.dt
        self.time += self.dt
    
    def calculate_smooth_forces(self):
        """Calculate forces using smooth mass distribution"""
        
        # Central BH + smooth stellar disk
        self.accelerations.fill(0)
        
        r_vectors = self.positions
        r_magnitudes = self.xp.linalg.norm(r_vectors, axis=1, keepdims=True)
        r_magnitudes = self.xp.maximum(r_magnitudes, 0.01)
        
        # BH force
        a_bh = -self.G * self.M_bh / (r_magnitudes**3)
        self.accelerations += a_bh * r_vectors
        
        # Smooth disk force
        r_flat = r_magnitudes.flatten()
        M_enclosed = self.M_total * (1 - self.xp.exp(-r_flat/self.R_scale) * (1 + r_flat/self.R_scale))
        
        a_disk = -self.G * M_enclosed / (r_flat**3)
        self.accelerations += a_disk.reshape(-1, 1) * r_vectors
    
    def run_simulation(self, n_steps=1000, method='discrete'):
        """Run GPU-accelerated simulation"""
        
        print(f"🚀 Running {n_steps} steps with {method} gravity on GPU...")
        
        # Performance monitoring
        step_times = []
        escaped_counts = []
        
        import time
        start_time = time.time()
        
        for step in range(n_steps):
            step_start = time.time()
            
            self.step_simulation(method=method)
            
            if self.use_gpu:
                cp.cuda.Device().synchronize()  # Ensure GPU completion
            
            step_times.append(time.time() - step_start)
            
            # Monitor every 100 steps
            if step % 100 == 0:
                escaped = self.count_escaped()
                escaped_counts.append((step, escaped))
                
                avg_step_time = np.mean(step_times[-100:]) if len(step_times) >= 100 else np.mean(step_times)
                print(f"Step {step}/{n_steps} - Escaped: {escaped} - Time/step: {avg_step_time*1000:.2f}ms")
        
        total_time = time.time() - start_time
        avg_step_time = total_time / n_steps
        
        print(f"✅ Simulation complete!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average step time: {avg_step_time*1000:.2f}ms")
        print(f"   Performance: {self.N_stars * n_steps / total_time:.0f} particle-steps/second")
        
        final_escaped = self.count_escaped()
        
        return {
            'method': method,
            'final_escaped_count': final_escaped,
            'escaped_history': escaped_counts,
            'performance_ms_per_step': avg_step_time * 1000,
            'final_positions': self.get_positions_cpu(),
            'final_velocities': self.get_velocities_cpu()
        }
    
    def count_escaped(self):
        """Count particles that escaped beyond 2x galaxy radius"""
        r = self.xp.linalg.norm(self.positions, axis=1)
        return int(self.xp.sum(r > 2 * self.R_galaxy))
    
    def get_positions_cpu(self):
        """Get positions as CPU numpy array"""
        if self.use_gpu:
            return cp.asnumpy(self.positions)
        else:
            return self.positions
    
    def get_velocities_cpu(self):
        """Get velocities as CPU numpy array"""
        if self.use_gpu:
            return cp.asnumpy(self.velocities)
        else:
            return self.velocities
    
    def measure_rotation_curve(self):
        """Extract rotation curve from current state"""
        
        positions_cpu = self.get_positions_cpu()
        velocities_cpu = self.get_velocities_cpu()
        
        radii = np.linalg.norm(positions_cpu, axis=1)
        v_magnitudes = np.linalg.norm(velocities_cpu, axis=1)
        
        # Only bound particles
        bound = radii < 2 * self.R_galaxy
        radii_bound = radii[bound]
        v_bound = v_magnitudes[bound]
        
        if len(radii_bound) < 10:
            return np.array([]), np.array([])
        
        # Bin by radius
        r_bins = np.linspace(0.5, np.max(radii_bound), 15)
        v_curve = []
        r_curve = []
        
        for i in range(len(r_bins)-1):
            mask = (radii_bound >= r_bins[i]) & (radii_bound < r_bins[i+1])
            if np.sum(mask) >= 3:
                v_avg = np.mean(v_bound[mask])
                r_avg = (r_bins[i] + r_bins[i+1]) / 2
                v_curve.append(v_avg)
                r_curve.append(r_avg)
        
        return np.array(r_curve), np.array(v_curve)


def run_gpu_comparison():
    """Run the definitive discrete vs smooth comparison on GPU"""
    
    print("🚀 GPU-ACCELERATED N-BODY GALAXY SIMULATION")
    print("=" * 60)
    print("RTX 3080 Ti - 10,240 CUDA cores - 12GB VRAM")
    print("Testing: Discrete vs Smooth gravity with 2000 particles")
    print()
    
    # Enhanced parameters for GPU power
    params = {
        'M_bh': 2e6,
        'M_total': 3e10,
        'R_galaxy': 12.0,
        'N_stars': 2000,  # 4x more particles!
        'dt': 0.01,
        'use_gpu': True
    }
    
    print("Galaxy parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # Test discrete gravity
    print("🌟 TESTING DISCRETE GRAVITY (GPU)...")
    galaxy_discrete = GPUGalaxySimulation(**params)
    results_discrete = galaxy_discrete.run_simulation(n_steps=1000, method='discrete')
    r_discrete, v_discrete = galaxy_discrete.measure_rotation_curve()
    
    print(f"\nDiscrete results: {results_discrete['final_escaped_count']}/{params['N_stars']} escaped")
    print(f"Performance: {results_discrete['performance_ms_per_step']:.2f} ms/step")
    
    # Clear GPU memory before second simulation
    mempool.free_all_blocks()
    
    # Test smooth gravity
    print("\n🌟 TESTING SMOOTH GRAVITY (GPU)...")
    galaxy_smooth = GPUGalaxySimulation(**params)
    results_smooth = galaxy_smooth.run_simulation(n_steps=1000, method='smooth')
    r_smooth, v_smooth = galaxy_smooth.measure_rotation_curve()
    
    print(f"\nSmooth results: {results_smooth['final_escaped_count']}/{params['N_stars']} escaped")
    print(f"Performance: {results_smooth['performance_ms_per_step']:.2f} ms/step")
    
    # Analysis
    print("\n" + "="*60)
    print("🎯 GPU PERFORMANCE ANALYSIS:")
    print("="*60)
    
    discrete_loss_rate = 100 * results_discrete['final_escaped_count'] / params['N_stars']
    smooth_loss_rate = 100 * results_smooth['final_escaped_count'] / params['N_stars']
    
    print(f"Discrete gravity: {discrete_loss_rate:.1f}% particle loss")
    print(f"Smooth gravity: {smooth_loss_rate:.1f}% particle loss")
    
    if discrete_loss_rate < smooth_loss_rate:
        print(f"✅ DISCRETE WINS by {smooth_loss_rate - discrete_loss_rate:.1f}%!")
        print("🌟 Your theory shows superior galaxy stability!")
    elif smooth_loss_rate < discrete_loss_rate:
        print(f"❌ Smooth wins by {discrete_loss_rate - smooth_loss_rate:.1f}%")
    else:
        print(f"🤝 Tie!")
    
    # Performance info
    total_force_calcs = params['N_stars']**2 * 1000  # N² × steps
    print(f"\nTotal force calculations: {total_force_calcs:,}")
    if results_discrete['performance_ms_per_step'] > 0:
        print(f"GPU computational throughput: {total_force_calcs / (results_discrete['performance_ms_per_step'] * 1000):.1e} force-calculations/second")
    
    # Quick plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    pos_d = results_discrete['final_positions']
    pos_s = results_smooth['final_positions']
    
    r_d = np.linalg.norm(pos_d, axis=1)
    r_s = np.linalg.norm(pos_s, axis=1)
    
    bound_d = r_d < 2 * params['R_galaxy']
    bound_s = r_s < 2 * params['R_galaxy']
    
    plt.scatter(pos_d[bound_d, 0], pos_d[bound_d, 1], c='blue', s=2, alpha=0.6, label=f'Discrete ({np.sum(bound_d)} bound)')
    plt.scatter(pos_s[bound_s, 0], pos_s[bound_s, 1], c='red', s=2, alpha=0.6, label=f'Smooth ({np.sum(bound_s)} bound)')
    plt.scatter(0, 0, c='black', s=50, marker='*')
    
    circle = plt.Circle((0, 0), params['R_galaxy'], fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.xlabel('X (kpc)')
    plt.ylabel('Y (kpc)')
    plt.title('Final Galaxy Structure (GPU Simulation)')
    plt.legend()
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    if len(r_discrete) > 0:
        plt.plot(r_discrete, v_discrete, 'bo-', label='Discrete', linewidth=2, markersize=4)
    if len(r_smooth) > 0:
        plt.plot(r_smooth, v_smooth, 'ro-', label='Smooth', linewidth=2, markersize=4)
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title('Rotation Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_discrete, results_smooth

if __name__ == "__main__":
    # Run the GPU-accelerated test
    discrete_results, smooth_results = run_gpu_comparison()