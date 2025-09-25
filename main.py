import numpy as np
import pandas as pd
import PyMieScatt as ps

# -------------------------
# 1. Simulation Parameters
# -------------------------
wavelength = 0.66  # µm (red laser, 660 nm typical for flow cytometry)
n_medium = 1.33    # refractive index of water

# Define angles (deg) for FSC, SSC, BSC
fsc_angle = 10    # forward scatter ~0–20°
ssc_angle = 90    # side scatter ~90°
bsc_angle = 170   # back scatter ~150–180°

# -------------------------
# 2. Helper function to simulate scatter
# -------------------------
def simulate_scatter(size_um, refr_index, noise_std=0.05):
    """
    Simulates FSC, SSC, BSC intensities for a given particle size and refractive index.
    Adds Gaussian noise to mimic real sensor data.
    """
    # Relative refractive index
    m = refr_index / n_medium

    # Size parameter x = π * d / λ
    x = np.pi * size_um / wavelength

    # Angles (cosine)
    cos_theta_fsc = np.cos(np.radians(fsc_angle))
    cos_theta_ssc = np.cos(np.radians(ssc_angle))
    cos_theta_bsc = np.cos(np.radians(bsc_angle))

    # Get scattering amplitudes S1, S2 at those angles
    S1_fsc, S2_fsc = ps.MieS1S2(m, x, cos_theta_fsc)
    S1_ssc, S2_ssc = ps.MieS1S2(m, x, cos_theta_ssc)
    S1_bsc, S2_bsc = ps.MieS1S2(m, x, cos_theta_bsc)

    # Intensity = |S1|^2 + |S2|^2
    fsc_int = (np.abs(S1_fsc)**2 + np.abs(S2_fsc)**2).real
    ssc_int = (np.abs(S1_ssc)**2 + np.abs(S2_ssc)**2).real
    bsc_int = (np.abs(S1_bsc)**2 + np.abs(S2_bsc)**2).real

    # Ratio feature
    ratio = fsc_int / (ssc_int + 1e-9)  # avoid divide by zero

    # Add Gaussian noise (multiplicative)
    fsc_int *= (1 + np.random.normal(0, noise_std))
    ssc_int *= (1 + np.random.normal(0, noise_std))
    bsc_int *= (1 + np.random.normal(0, noise_std))
    ratio = fsc_int / (ssc_int + 1e-9)

    # Noise level stored separately (what was applied)
    noise_level = noise_std

    return fsc_int, ssc_int, bsc_int, ratio, refr_index, noise_level, size_um

# -------------------------
# 3. Generate dataset
# -------------------------
rows = []
for _ in range(10000):
    # Random particle size between 1 and 20 µm
    size_um = np.random.uniform(1, 20)
    # Random refractive index between 1.50 and 1.60
    refr_index = np.random.uniform(1.50, 1.60)

    row = simulate_scatter(size_um, refr_index, noise_std=0.05)
    rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows, columns=[
    "fsc_peak",
    "ssc_peak",
    "bsc_peak",
    "fsc_ssc_ratio",
    "refractive_index",
    "noise_level",
    "size_um"
])

# -------------------------
# 4. Save and display
# -------------------------
print(df.head(10))
df.to_csv("synthetic_scatter_data.csv", index=False)
print("\n✅ Saved 100-row synthetic dataset to synthetic_scatter_data.csv")
