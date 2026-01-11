import numpy as np
import matplotlib.pyplot as plt
import warnings

# ------------------------------------------------------------------
# Transfer Matrix Method (TMM) for OBLIQUE INCIDENCE
# ------------------------------------------------------------------

def layer_matrix_oblique(n, d, wavelength, theta0, n0, polarization):
    """
    Returns transfer matrix for a single layer at oblique incidence.
    n: refractive index of the layer
    d: thickness of the layer (in nm)
    wavelength: free-space wavelength (in nm)
    theta0: angle of incidence in the initial medium (radians)
    n0: refractive index of the initial medium
    polarization: 'TE' or 'TM'
    """
    # Calculate angle inside the layer using Snell's Law.
    # We use a complex-safe sqrt to handle Total Internal Reflection (TIR)
    sin_theta = (n0 / n) * np.sin(theta0)
    cos_theta = np.sqrt(1 - sin_theta**2, dtype=complex)

    # Calculate phase thickness (delta)
    k0 = 2 * np.pi / wavelength
    delta = k0 * n * d * cos_theta

    # Calculate optical admittance (eta) based on polarization
    if polarization == 'TE':
        eta = n * cos_theta
    elif polarization == 'TM':
        eta = n / cos_theta
    else:
        raise ValueError("Polarization must be 'TE' or 'TM'")

    # Construct the transfer matrix M
    m11 = np.cos(delta)
    m12 = 1j * np.sin(delta) / eta
    m21 = 1j * eta * np.sin(delta)
    m22 = np.cos(delta)
    return np.array([[m11, m12], [m21, m22]], dtype=complex)

def multilayer_transfer_matrix_oblique(n_list, d_list, wavelength, theta0, n0, polarization):
    """
    Calculates the total transfer matrix for a multilayer stack at oblique incidence.
    """
    M = np.identity(2, dtype=complex)
    for n, d in zip(n_list, d_list):
        M = np.matmul(M, layer_matrix_oblique(n, d, wavelength, theta0, n0, polarization))
    return M

def transmission_oblique(n0, ns, n_list, d_list, wavelengths, theta0_deg, polarization):
    """
    Computes the transmission spectrum for a given angle and polarization.
    theta0_deg: angle of incidence in degrees
    """
    T = []
    theta0_rad = np.deg2rad(theta0_deg)

    # Calculate admittances for incident and substrate media
    sin_theta_s = (n0 / ns) * np.sin(theta0_rad)
    cos_theta_s = np.sqrt(1 - sin_theta_s**2, dtype=complex)
    cos_theta_0 = np.cos(theta0_rad)

    if polarization == 'TE':
        eta0 = n0 * cos_theta_0
        etas = ns * cos_theta_s
    elif polarization == 'TM':
        eta0 = n0 / cos_theta_0
        etas = ns / cos_theta_s
    else:
        raise ValueError("Polarization must be 'TE' or 'TM'")

    for wl in wavelengths:
        M = multilayer_transfer_matrix_oblique(n_list, d_list, wl, theta0_rad, n0, polarization)
        m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]

        # Calculate the amplitude transmission coefficient (t)
        denom = (m11 + m12*etas)*eta0 + (m21 + m22*etas)
        t_amp = 2 * eta0 / denom

        # Power Transmittance (Transmissivity)
        transmittance = (etas/eta0).real * np.abs(t_amp)**2
        T.append(transmittance)

    return np.array(T)

# ------------------------------------------------------------------
# Structure Definition (Unchanged)
# ------------------------------------------------------------------
n_TiO2 = 2.5197
n_SiO2 = 1.4533
n_defect = 1.4634
n_air = 1.0

d_TiO2 = 800 / (4 * n_TiO2)
d_SiO2 = 800 / (4 * n_SiO2)
d_defect = 800 / (2 * n_defect)

N = 6
n_list = []
d_list = []
for _ in range(N):
    n_list.extend([n_TiO2, n_SiO2])
    d_list.extend([d_TiO2, d_SiO2])
n_list.append(n_defect)
d_list.append(d_defect)
for _ in range(N):
    n_list.extend([n_TiO2, n_SiO2])
    d_list.extend([d_TiO2, d_SiO2])

# ------------------------------------------------------------------
# Simulation: Loop over angles and find resonance peaks
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Simulation: Loop over angles and TRACK the resonance peak
# ------------------------------------------------------------------

wavelengths = np.linspace(600, 1000, 800)  # Wavelength range in nm
angles = np.linspace(0, 80, 50)           # Angle range in degrees

peak_wavelengths_te = []
peak_wavelengths_tm = []

# --- Initial peak location and search window settings ---
# Start by looking for the peak around its design wavelength (800 nm)
expected_wl_te = 935
expected_wl_tm = 935
# Define how wide the search window should be (in nm)
search_window_nm = 30

print("Calculating...")
for angle in angles:
    # --- TE Polarization ---
    T_te = transmission_oblique(n_air, n_air, n_list, d_list, wavelengths, angle, 'TE')
    # Create a boolean mask to define the search window for TE
    te_mask = (wavelengths > expected_wl_te - search_window_nm / 2) & \
              (wavelengths < expected_wl_te + search_window_nm / 2)
    # Find the index of the max peak *only within the masked window*
    peak_index_in_window = np.argmax(T_te[te_mask])
    # Get the actual wavelength value of that peak
    found_wl_te = wavelengths[te_mask][peak_index_in_window]
    peak_wavelengths_te.append(found_wl_te)
    # IMPORTANT: Update the expected wavelength for the next iteration
    expected_wl_te = found_wl_te

    # --- TM Polarization (repeat the same logic) ---
    T_tm = transmission_oblique(n_air, n_air, n_list, d_list, wavelengths, angle, 'TM')
    # Create a boolean mask to define the search window for TM
    tm_mask = (wavelengths > expected_wl_tm - search_window_nm / 2) & \
              (wavelengths < expected_wl_tm + search_window_nm / 2)
    # Find the peak within the masked window
    peak_index_in_window = np.argmax(T_tm[tm_mask])
    # Get the wavelength value
    found_wl_tm = wavelengths[tm_mask][peak_index_in_window]
    peak_wavelengths_tm.append(found_wl_tm)
    # Update the expected wavelength for the next iteration
    expected_wl_tm = found_wl_tm

print("Calculation complete.")

# ------------------------------------------------------------------
# Plotting (Code is unchanged)
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Resonance Peak vs. Incidence Angle', fontsize=16)

# --- TE Plot ---
ax1.plot(angles, peak_wavelengths_te, 'o-', color='royalblue', label='TE Mode')
ax1.set_title('TE Polarization')
ax1.set_xlabel("Incidence Angle (degrees)")
ax1.set_ylabel("Resonance Wavelength (nm)")
ax1.grid(True)
ax1.legend()

# --- TM Plot ---
ax2.plot(angles, peak_wavelengths_tm, 'o-', color='crimson', label='TM Mode')
ax2.set_title('TM Polarization')
ax2.set_xlabel("Incidence Angle (degrees)")
ax2.set_ylabel("Resonance Wavelength (nm)")
ax2.grid(True)
ax2.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()