import numpy as np
import matplotlib.pyplot as plt

# This code simulates a 1D photonic crystal with a defect layer,
# a structure often used in optical sensors, filters, and lasers.
# This version uses Bruggeman's Effective Medium Theory (EMT) to model
# how the defect layer's refractive index changes with atrazine concentration.

# --------------------------------------------------------------------
# Transfer Matrix Method (TMM) Functions
# The TMM is a computational method used to analyze the propagation
# of electromagnetic waves through a layered medium.
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
    Sum[ fraction_i * (n_i^2 - n_eff^2) / (n_i^2 + 2*n_eff^2) ] = 0
    """
    total = 0
    n_eff_sq = n_eff**2
    for f, n in zip(fractions, n_components):
        n_sq = n**2
        total += f * (n_sq - n_eff_sq) / (n_sq + 2 * n_eff_sq)
    return total

def solve_bruggeman_n_eff(fractions, n_components, initial_guess=1.4):
    """
    Solves for the effective refractive index (n_eff) using a simple
    numerical root-finding algorithm (bisection method).
    """
    low = min(n_components)
    high = max(n_components)
    
    if bruggeman_equation(low, fractions, n_components) * bruggeman_equation(high, fractions, n_components) >= 0:
        return np.sum(np.array(fractions) * np.array(n_components))

    for _ in range(100):
        mid = (low + high) / 2
        if mid == low or mid == high:
            return mid
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

n_mip_polymer = 1.49
n_atrazine = 1.6
n_defect_initial = 1.4634

f_voids_initial_guess = 0.1
f_polymer = 1 - f_voids_initial_guess
n_eff_calc = solve_bruggeman_n_eff([f_polymer, f_voids_initial_guess], [n_mip_polymer, n_air])

f_voids = f_voids_initial_guess
for _ in range(50):
    n_calc = solve_bruggeman_n_eff([1-f_voids, f_voids], [n_mip_polymer, n_air])
    if abs(n_calc - n_defect_initial) < 1e-6:
        break
    f_voids -= (n_calc - n_defect_initial) * 0.1
f_mip_polymer = 1.0 - f_voids
print(f"Calculated MIP porosity (f_voids): {f_voids:.4f}")

d_TiO2 = 800 / (4 * n_TiO2)
d_SiO2 = 800 / (4 * n_SiO2)
d_defect = 800 / (2 * n_defect_initial)

N = 8

# --------------------------------------------------------------------
# Simulation and Plotting
# --------------------------------------------------------------------
wavelengths = np.linspace(900, 1000, 1000)

atrazine_conc_uM = np.logspace(-3, 1, 50)
K_D_uM = 11e-3
theta = atrazine_conc_uM / (K_D_uM + atrazine_conc_uM)

n_defect_array = []
print("Calculating n_eff using Bruggeman's model for each concentration...")
for t in theta:
    f_filled_voids = t * f_voids
    f_unfilled_voids = (1 - t) * f_voids
    fractions = [f_mip_polymer, f_unfilled_voids, f_filled_voids]
    n_components = [n_mip_polymer, n_air, n_atrazine]
    n_eff = solve_bruggeman_n_eff(fractions, n_components)
    n_defect_array.append(n_eff)

resonance_peaks = []
print("Simulating transmission for each refractive index...")

for n_defect_current in n_defect_array:
    n_list = []
    d_list = []
    for _ in range(N):
        n_list.extend([n_TiO2, n_SiO2])
        d_list.extend([d_TiO2, d_SiO2])
    n_list.append(n_defect_current)
    d_list.append(d_defect)
    for _ in range(N):
        n_list.extend([n_TiO2, n_SiO2])
        d_list.extend([d_TiO2, d_SiO2])

    T = transmission(n_air, n_air, n_list, d_list, wavelengths)

    indices_in_stop_band = np.where(T < 0.5)[0]
    peak_wavelength = np.nan

    if len(indices_in_stop_band) > 0:
        start_index = indices_in_stop_band[0]
        end_index = indices_in_stop_band[-1]
        search_slice = T[start_index:end_index+1]
        
        if len(search_slice) > 0:
            peak_index_in_slice = np.argmax(search_slice)
            peak_index = start_index + peak_index_in_slice
            peak_wavelength = wavelengths[peak_index]
    else:
        print("Warning: Could not identify a stop band below 50% transmission. Falling back to global peak search.")
        peak_index = np.argmax(T)
        peak_wavelength = wavelengths[peak_index]

    resonance_peaks.append(peak_wavelength)

print("Simulation complete.")


# ----------------------------------------------------------------
# --- Sensitivity Calculation and Plotting ---
# ----------------------------------------------------------------

# Calculate the change in wavelength and concentration
delta_wavelength = np.diff(resonance_peaks)
delta_concentration = np.diff(atrazine_conc_uM)

# Calculate sensitivity (dW/dC)
sensitivity = delta_wavelength / delta_concentration

# The sensitivity corresponds to the midpoint between concentration points
concentration_midpoints = (atrazine_conc_uM[:-1] + atrazine_conc_uM[1:]) / 2

# --- Plot Sensitivity vs. Concentration ---
plt.figure(figsize=(8, 6))
plt.plot(concentration_midpoints, sensitivity, 'ro-', markersize=5, linewidth=2)
plt.xscale('log')
plt.xlabel("Atrazine Concentration (µM)", fontsize=14)
plt.ylabel("Sensitivity (nm/µM)", fontsize=14)
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

