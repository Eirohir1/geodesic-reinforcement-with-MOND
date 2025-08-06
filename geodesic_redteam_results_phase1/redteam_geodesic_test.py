# redteam_geodesic_test.py

import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math
import json

# === Physical Constants ===
G = 6.67430e-11
CUTOFF_PC = 0.05
CUTOFF_SI = CUTOFF_PC * 3.086e16
GALAXY_RADIUS = 1.5e20  # 50 kpc
NUM_PARTICLES = 6000
NUM_STEPS = 200
DT = 1e10
THREADS_PER_BLOCK = 256
BLOCKS = (NUM_PARTICLES + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

# === CUDA Kernel ===
@cuda.jit
def redteam_kernel(pos, mass, acc, galaxy_radius, cutoff_radius, N, G_const, mode):
    i = cuda.grid(1)
    if i < N:
        ax = 0.0
        ay = 0.0
        az = 0.0
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
        for j in range(N):
            if i != j:
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                dz = zi - pos[j, 2]
                dist_sq = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist_sq + 1e14)

                force = 0.0
                if mode == 0:  # Newtonian
                    if dist > 1e-2:
                        force = -G_const * mass[j] / (dist_sq * dist)
                elif mode == 1:  # Cutoff only
                    if dist < cutoff_radius:
                        force = -G_const * mass[j] / (dist_sq * dist)
                elif mode == 2:  # Cutoff + taper
                    if dist < cutoff_radius:
                        taper = 1.0 / (1.0 + dist * dist / (galaxy_radius * galaxy_radius * 0.01))
                        force = -G_const * mass[j] * taper / (dist_sq * dist)

                ax += force * dx
                ay += force * dy
                az += force * dz

        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az

def generate_galaxy():
    masses = np.random.lognormal(np.log(0.5 * 1.989e30), 0.8, NUM_PARTICLES)
    masses[0] = 8e36  # CBH

    scale_radius = GALAXY_RADIUS / 4
    radii = np.random.exponential(scale_radius, NUM_PARTICLES)
    radii = np.clip(radii, 5e17, GALAXY_RADIUS)
    angles = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
    z_pos = np.random.normal(0, 2e17, NUM_PARTICLES)

    pos = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
    pos[:, 0] = radii * np.cos(angles)
    pos[:, 1] = radii * np.sin(angles)
    pos[:, 2] = z_pos

    vel = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
    for i in range(1, NUM_PARTICLES):
        r = radii[i]
        enclosed_mass = masses[0] + np.sum(masses[radii < r])
        v_circ = np.sqrt(G * enclosed_mass / r) if r > 0 else 0
        vel[i, 0] = -v_circ * np.sin(angles[i])
        vel[i, 1] = v_circ * np.cos(angles[i])
        vel[i] += np.random.normal(0, v_circ * 0.1, 3)

    return pos, vel, masses

def run_mode(mode, label):
    pos, vel, mass = generate_galaxy()
    pos_gpu = cp.array(pos)
    vel_gpu = cp.array(vel)
    mass_gpu = cp.array(mass)
    acc_gpu = cp.zeros((NUM_PARTICLES, 3), dtype=cp.float32)

    for step in range(NUM_STEPS):
        redteam_kernel[BLOCKS, THREADS_PER_BLOCK](pos_gpu, mass_gpu, acc_gpu,
                                                  GALAXY_RADIUS, CUTOFF_SI, NUM_PARTICLES,
                                                  G, mode)
        cp.cuda.Device().synchronize()
        vel_gpu += acc_gpu * DT
        pos_gpu += vel_gpu * DT

    final_pos = cp.asnumpy(pos_gpu)
    final_vel = cp.asnumpy(vel_gpu)

    # Metrics
    radii = np.sqrt(final_pos[:, 0] ** 2 + final_pos[:, 1] ** 2)
    bound = np.sum(radii < 2 * GALAXY_RADIUS)
    retention = 100 * bound / NUM_PARTICLES

    vmag = np.linalg.norm(final_vel[1:], axis=1)
    dispersion = np.std(vmag) / 1000  # km/s

    Lz = final_pos[:, 0] * final_vel[:, 1] - final_pos[:, 1] * final_vel[:, 0]

    return final_pos, final_vel, Lz, retention, dispersion

def plot_all(pos, vel, Lz, label, outdir):
    kpc = 3.086e19
    pos_kpc = pos / kpc

    # Plot 1: Galaxy structure
    plt.figure()
    plt.scatter(pos_kpc[1:, 0], pos_kpc[1:, 1], s=0.5, alpha=0.7, color='white')
    plt.scatter(pos_kpc[0, 0], pos_kpc[0, 1], s=100, color='red', marker='*')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.title(f"Galaxy Snapshot - {label}")
    plt.xlabel("X (kpc)")
    plt.ylabel("Y (kpc)")
    plt.gca().set_facecolor('black')
    plt.savefig(f"{outdir}/{label}_galaxy.png", dpi=300)
    plt.close()

    # Plot 2: Rotation curve
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    v_tang = np.abs(pos[:, 0]*vel[:, 1] - pos[:, 1]*vel[:, 0]) / r
    bins = np.linspace(0, GALAXY_RADIUS, 50)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    vr_means = [np.mean(v_tang[(r >= bins[i]) & (r < bins[i+1])]) / 1000 for i in range(len(bin_centers))]

    plt.figure()
    plt.plot(bin_centers / 3.086e16, vr_means, 'o-', color='blue')
    plt.title(f"Rotation Curve - {label}")
    plt.xlabel("Radius (pc)")
    plt.ylabel("v_tangent (km/s)")
    plt.grid(True)
    plt.savefig(f"{outdir}/{label}_rotation.png", dpi=300)
    plt.close()

    # Plot 3: Angular Momentum
    plt.figure()
    plt.hist(Lz[1:], bins=100, color='cyan', alpha=0.8)
    plt.title(f"Lz Histogram - {label}")
    plt.xlabel("Lz")
    plt.ylabel("Count")
    plt.savefig(f"{outdir}/{label}_lz_histogram.png", dpi=300)
    plt.close()

def main():
    os.makedirs("geodesic_redteam_results", exist_ok=True)
    summary = []

    for mode, label in zip([0, 1, 2], ["mode_a", "mode_b", "mode_c"]):
        print(f"\n🚀 Running {label.upper()}...")
        pos, vel, Lz, retention, dispersion = run_mode(mode, label)
        summary.append((label, retention, dispersion))
        plot_all(pos, vel, Lz, label, "geodesic_redteam_results")

    with open("geodesic_redteam_results/summary.txt", "w") as f:
        for label, r, d in summary:
            f.write(f"{label}: Retention = {r:.2f}%, Dispersion = {d:.2f} km/s\n")

    print("\n✅ All modes completed. See 'geodesic_redteam_results/' for output.")

if __name__ == "__main__":
    main()
