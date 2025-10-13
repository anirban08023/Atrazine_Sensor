import numpy as np
import matplotlib.pyplot as plt
import csv

# This code simulates a 1D photonic crystal with a defect layer.
# This OPTIMIZED version vectorizes the TMM calculations using NumPy
# to process all wavelengths simultaneously, resulting in a significant
# speedup on the CPU.

# --- Parameters for Oblique Incidence ---
ANGLE_DEG = 80 # Angle of incidence in degrees
POLARIZATION = 's' # 's' for TE, 'p' for TM

# --------------------------------------------------------------------
# VECTORIZED Transfer Matrix Method (TMM) Functions for Oblique Incidence
# --------------------------------------------------------------------

def vectorized_layer_matrix(n, d, wavelengths, n0, theta0):
    """
    Calculates the transfer matrices for a single layer across all wavelengths at once.
    """
    # Reshape wavelengths array for broadcasting
    # Shape: (num_wavelengths, 1, 1) to allow element-wise ops with (2,2) matrices
    wl = wavelengths[:, np.newaxis, np.newaxis]
    k0 = 2 * np.pi / wl

    # Snell's Law (complex arcsin for TIR)
    sin_theta = (n0 / n) * np.sin(theta0)
    theta = np.arcsin(sin_theta)
    cos_theta = np.cos(theta)
    
    delta = k0 * n * d * cos_theta
    
    if POLARIZATION.lower() == 's':
        eta = n * cos_theta
    elif POLARIZATION.lower() == 'p':
        eta = n / cos_theta
    else:
        raise ValueError("Polarization must be 's' or 'p'")

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    # Build the matrix for all wavelengths simultaneously
    # Each operation is on an array of shape (num_wavelengths, 1, 1)
    m11 = cos_delta
    m12 = 1j * sin_delta / eta
    m21 = 1j * eta * sin_delta
    m22 = cos_delta
    
    # Stack into a (num_wavelengths, 2, 2) array
    M = np.concatenate((np.concatenate((m11, m12), axis=2),
                        np.concatenate((m21, m22), axis=2)), axis=1)
    return M

def vectorized_multilayer_transfer_matrix(n_list, d_list, wavelengths, n0, theta0):
    """
    Calculates the total transfer matrix for the stack across all wavelengths.
    """
    # Initialize a stack of identity matrices, one for each wavelength
    num_wavelengths = len(wavelengths)
    M = np.broadcast_to(np.identity(2, dtype=complex), (num_wavelengths, 2, 2)).copy()
    
    # Sequentially multiply by each layer's matrix stack
    for n, d in zip(n_list, d_list):
        layer_M = vectorized_layer_matrix(n, d, wavelengths, n0, theta0)
        M = np.matmul(M, layer_M) # np.matmul handles stacking automatically
    return M

def vectorized_transmission(n0, ns, n_list, d_list, wavelengths, theta0):
    """
    Computes the transmission spectrum for the multilayer stack, fully vectorized.
    """
    M = vectorized_multilayer_transfer_matrix(n_list, d_list, wavelengths, n0, theta0)
    
    sin_thetas = (n0 / ns) * np.sin(theta0)
    thetas = np.arcsin(sin_thetas)

    if POLARIZATION.lower() == 's':
        eta0 = n0 * np.cos(theta0)
        etas = ns * np.cos(thetas)
    else: # 'p' polarization
        eta0 = n0 / np.cos(theta0)
        etas = ns / np.cos(thetas)
            
    # Extract components from the stack of matrices M (shape: num_wavelengths, 2, 2)
    m11 = M[:, 0, 0]
    m12 = M[:, 0, 1]
    m21 = M[:, 1, 0]
    m22 = M[:, 1, 1]
    
    denom = (m11 + m12 * etas) * eta0 + (m21 + m22 * etas)
    t = 2 * eta0 / denom
    
    T = ((etas / eta0).real) * abs(t)**2
    return T

# --------------------------------------------------------------------
# Bruggeman's Effective Medium Theory (EMT) Section (Unchanged)
# --------------------------------------------------------------------

def bruggeman_equation(n_eff, fractions, n_components):
    total = 0
    n_eff_sq = n_eff**2
    for f, n in zip(fractions, n_components):
        n_sq = n**2
        total += f * (n_sq - n_eff_sq) / (n_sq + 2 * n_eff_sq)
    return total

def solve_bruggeman_n_eff(fractions, n_components):
    low = min(n_components)
    high = max(n_components)
    if bruggeman_equation(low, fractions, n_components) * bruggeman_equation(high, fractions, n_components) >= 0:
        return np.sum(np.array(fractions) * np.array(n_components))
    for _ in range(100):
        mid = (low + high) / 2
        if mid == low or mid == high: return mid
        y_mid = bruggeman_equation(mid, fractions, n_components)
        y_low = bruggeman_equation(low, fractions, n_components)
        if y_low * y_mid < 0: high = mid
        else: low = mid
    return (low + high) / 2

# --------------------------------------------------------------------
# Structure Definition (Unchanged)
# --------------------------------------------------------------------
n_TiO2 = 2.5197
n_SiO2 = 1.4533
n_air = 1.0
n_mip_polymer = 1.49
n_atrazine = 1.6
n_defect_initial = 1.4634

f_voids = 0.1
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
# Simulation and Plotting (Updated to use vectorized functions)
# --------------------------------------------------------------------
wavelengths = np.linspace(670, 825, 500000)
atrazine_conc_uM = np.logspace(-3, 1, 50)
K_D_uM = 11e-3
theta = atrazine_conc_uM / (K_D_uM + atrazine_conc_uM)

n_defect_array = []
print("Calculating n_eff using Bruggeman's model...")
for t in theta:
    fractions = [f_mip_polymer, (1 - t) * f_voids, t * f_voids]
    n_components = [n_mip_polymer, n_air, n_atrazine]
    n_defect_array.append(solve_bruggeman_n_eff(fractions, n_components))

resonance_peaks = []
T_lowest_conc, T_highest_conc = None, None
print("Simulating transmission for each refractive index...")
theta0_rad = np.deg2rad(ANGLE_DEG)

for i, n_defect_current in enumerate(n_defect_array):
    n_list = [n_TiO2, n_SiO2] * N + [n_defect_current] + [n_TiO2, n_SiO2] * N
    d_list = [d_TiO2, d_SiO2] * N + [d_defect] + [d_TiO2, d_SiO2] * N

    # --- CALL THE OPTIMIZED VECTORIZED FUNCTION ---
    T = vectorized_transmission(n_air, n_air, n_list, d_list, wavelengths, theta0_rad)
    
    if i == 0: T_lowest_conc = T
    if i == len(n_defect_array) - 1: T_highest_conc = T

    indices_in_stop_band = np.where(T < 0.5)[0]
    peak_wavelength = np.nan
    if len(indices_in_stop_band) > 0:
        start_index, end_index = indices_in_stop_band[0], indices_in_stop_band[-1]
        search_slice = T[start_index:end_index+1]
        if len(search_slice) > 0:
            peak_index = start_index + np.argmax(search_slice)
            peak_wavelength = wavelengths[peak_index]
    else:
        print("Warning: Could not identify a stop band. Falling back to global peak search.")
        peak_wavelength = wavelengths[np.argmax(T)]
    resonance_peaks.append(peak_wavelength)

print("Simulation complete.")

# ----------------------------------------------------------------
# --- Sensitivity Calculation and Plotting (Unchanged) ---
# ----------------------------------------------------------------
delta_wavelength = np.diff(resonance_peaks)
delta_concentration = np.diff(atrazine_conc_uM)
sensitivity = delta_wavelength / delta_concentration
concentration_midpoints = (atrazine_conc_uM[:-1] + atrazine_conc_uM[1:]) / 2

# ----------------------------------------------------------------
# --- CSV File Generation ---
# ----------------------------------------------------------------
csv_filename = 'sensitivity_data.csv'
print(f"Generating {csv_filename}...")
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write Header
    csv_writer.writerow(['Angle (deg)', 'Sensitivity (nm/uM)', 'Concentration (uM)'])
    # Write Data
    for sens, conc in zip(sensitivity, concentration_midpoints):
        csv_writer.writerow([ANGLE_DEG, sens, conc])
print("CSV file generated successfully.")


plt.figure(figsize=(8, 6))
plt.plot(concentration_midpoints, sensitivity, 'ro-', markersize=5, linewidth=2)
plt.xscale('log')
plt.xlabel("Atrazine Concentration (µM)", fontsize=14)
plt.ylabel("Sensitivity (nm/µM)", fontsize=14)
plt.title(f"Sensor Sensitivity at {ANGLE_DEG}° Incidence ({POLARIZATION.upper()}-pol)", fontsize=16)
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.tight_layout()

plt.figure(figsize=(8, 6))
if T_lowest_conc is not None and T_highest_conc is not None:
    plt.plot(wavelengths, T_lowest_conc, lw=2, label=f'Lowest Conc ({atrazine_conc_uM[0]:.1e} µM)')
    plt.plot(wavelengths, T_highest_conc, lw=2, label=f'Highest Conc ({atrazine_conc_uM[-1]:.1e} µM)')
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Transmittance", fontsize=14)
plt.title(f"Resonance Shift Range at {ANGLE_DEG}° ({POLARIZATION.upper()}-pol)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

