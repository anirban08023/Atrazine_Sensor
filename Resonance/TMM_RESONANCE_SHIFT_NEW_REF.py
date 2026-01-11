import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# This code simulates a 1D photonic crystal sensor for atrazine.
# It uses Bruggeman's Effective Medium Theory (EMT) combined with a
# Langmuir binding model to calculate the change in the defect layer's
# refractive index based on atrazine concentration.
# The final output is a plot of the resonance wavelength shift vs. concentration.

# --------------------------------------------------------------------
# Transfer Matrix Method (TMM) Functions
# --------------------------------------------------------------------

def layer_matrix(n, d, wavelength):
    """
    Calculates the transfer matrix for a single dielectric layer.
    """
    k0 = 2 * np.pi / wavelength
    delta = k0 * n * d
    m11 = np.cos(delta)
    m12 = 1j * np.sin(delta) / n
    m21 = 1j * n * np.sin(delta)
    m22 = np.cos(delta)
    return np.array([[m11, m12], [m21, m22]], dtype=complex)

def multilayer_transfer_matrix(n_list, d_list, wavelength):
    """
    Calculates the total transfer matrix for a multilayer stack.
    """
    M = np.identity(2, dtype=complex)
    for n, d in zip(n_list, d_list):
        M = np.matmul(M, layer_matrix(n, d, wavelength))
    return M

def transmission(n0, ns, n_list, d_list, wavelengths):
    """
    Computes the transmission spectrum for the multilayer stack.
    """
    T = []
    for wl in wavelengths:
        M = multilayer_transfer_matrix(n_list, d_list, wl)
        m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        denom = (m11 + m12 * ns) * n0 + (m21 + m22 * ns)
        t = 2 * n0 / denom
        T.append(abs(t)**2 * (ns / n0).real)
    return np.array(T)

# --------------------------------------------------------------------
# Bruggeman's Effective Medium Theory (EMT) Section
# --------------------------------------------------------------------

def bruggeman_equation(n_eff, fractions, n_components):
    """
    The core Bruggeman's EMT equation. We need to find the root of this equation.
    """
    total = 0
    n_eff_sq = n_eff**2
    for f, n in zip(fractions, n_components):
        n_sq = n**2
        total += f * (n_sq - n_eff_sq) / (n_sq + 2 * n_eff_sq)
    return total

def solve_bruggeman_n_eff(fractions, n_components):
    """
    Solves for the effective refractive index (n_eff) using a bisection method.
    """
    low = min(n_components)
    high = max(n_components)
    
    if bruggeman_equation(low, fractions, n_components) * bruggeman_equation(high, fractions, n_components) >= 0:
         return np.sum(np.array(fractions) * np.array(n_components))

    for _ in range(100):
        mid = (low + high) / 2
        if mid == low or mid == high: return mid
        y_mid = bruggeman_equation(mid, fractions, n_components)
        y_low = bruggeman_equation(low, fractions, n_components)
        if y_low * y_mid < 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2

# --------------------------------------------------------------------
# Structure Definition
# --------------------------------------------------------------------
n_TiO2 = 2.5197
n_SiO2 = 1.4533
n_air = 1.0

n_mip_polymer = 1.50
n_atrazine = 1.58
n_defect_initial = 1.4634

# Calculate initial porosity (f_voids) of the MIP layer
f_voids_initial_guess = 0.1
f_voids = f_voids_initial_guess
for _ in range(50):
    n_calc = solve_bruggeman_n_eff([1-f_voids, f_voids], [n_mip_polymer, n_air])
    if abs(n_calc - n_defect_initial) < 1e-6: break
    f_voids -= (n_calc - n_defect_initial) * 0.1
f_mip_polymer = 1.0 - f_voids
print(f"Calculated MIP porosity (f_voids): {f_voids:.4f}")

d_TiO2 = 800 / (4 * n_TiO2)
d_SiO2 = 800 / (4 * n_SiO2)
d_defect = 800 / (2 * n_defect_initial)
N = 8

# --------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------
wavelengths = np.linspace(900, 1000, 2000)
atrazine_conc_uM = np.logspace(-3, 1, 50) # Use 50 points for a smooth curve
K_D_uM = 11e-3
theta = atrazine_conc_uM / (K_D_uM + atrazine_conc_uM)

n_defect_array = []
print("\nCalculating n_eff using Bruggeman's model for each concentration...")
for t in theta:
    f_filled_voids = t * f_voids
    f_unfilled_voids = (1 - t) * f_voids
    fractions = [f_mip_polymer, f_unfilled_voids, f_filled_voids]
    n_components = [n_mip_polymer, n_air, n_atrazine]
    n_eff = solve_bruggeman_n_eff(fractions, n_components)
    n_defect_array.append(n_eff)

resonance_peaks = []
print("Simulating transmission and finding peaks for each refractive index...")

# Build base structure
n_base = []
d_base = []
for _ in range(N):
    n_base.extend([n_TiO2, n_SiO2])
    d_base.extend([d_TiO2, d_SiO2])

for n_defect_current in n_defect_array:
    n_stack = n_base + [n_defect_current] + n_base
    d_stack = d_base + [d_defect] + d_base
    
    T = transmission(n_air, n_air, n_stack, d_stack, wavelengths)
    
    # Find the resonance peak using find_peaks
    peaks, _ = find_peaks(T, height=0.5) # height threshold filters noise
    if len(peaks) > 0:
        # Of the found peaks, choose the one with the highest transmission
        peak_wavelength = wavelengths[peaks[np.argmax(T[peaks])]]
        resonance_peaks.append(peak_wavelength)
    else:
        resonance_peaks.append(np.nan) # Handle cases where no peak is found

print("Simulation complete.")

# --------------------------------------------------------------------
# Final Plotting and Analysis
# --------------------------------------------------------------------

# --- Calculate Wavelength Shift ---
delta_lambda = np.array(resonance_peaks) - resonance_peaks[0]

# --- Plot: Wavelength Shift vs. Atrazine Concentration ---
plt.figure(figsize=(8, 8))
plt.plot(atrazine_conc_uM, delta_lambda, 'mo-', markersize=6, linewidth=2)
plt.xscale('log')
plt.xlabel("Atrazine Concentration (µM)", fontsize=20)
plt.ylabel("Resonance Wavelength Shift (Δλ nm)", fontsize=20)
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.show()


# --- Print Numerical Results for reference ---
delta_n = np.array(n_defect_array) - n_defect_array[0]
print("\n--- Sensor Response Data ---")
print("Conc (µM) |   Δn    |  Δλ (nm)")
print("--------------------------------")
for c, dn, dl in zip(atrazine_conc_uM, delta_n, delta_lambda):
    print(f"{c:<9.4f} | {dn:+.4f} | {dl:+.2f}")

