import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Transfer Matrix Method (TMM) for Oblique Incidence
# -----------------------------

def layer_matrix(n, d, wavelength, n0, theta0, polarization='TE'):
    """
    Returns transfer matrix for a single dielectric layer at oblique incidence.
    n: refractive index of the layer
    d: thickness of the layer (in nm)
    wavelength: free-space wavelength (in nm)
    n0: refractive index of the incident medium (e.g., air)
    theta0: angle of incidence in the initial medium (in radians)
    polarization: 'TE' (s-pol) or 'TM' (p-pol)
    """
    k0 = 2 * np.pi / wavelength

    # --- MODIFICATION: Snell's Law ---
    # Calculate the angle inside the layer
    # np.arcsin can handle complex arguments for evanescent waves
    theta = np.arcsin((n0 / n) * np.sin(theta0))

    # --- MODIFICATION: Update phase thickness and effective index ---
    # The phase change now depends on the vertical component of the wavevector
    delta = k0 * n * d * np.cos(theta)
    
    # The effective refractive index (admittance) depends on polarization
    if polarization.upper() == 'TE':
        eta = n * np.cos(theta)  # Effective index for TE (s-polarization)
    elif polarization.upper() == 'TM':
        eta = n / np.cos(theta)  # Effective index for TM (p-polarization)
    else:
        raise ValueError("Polarization must be 'TE' or 'TM'")

    m11 = np.cos(delta)
    m12 = 1j * np.sin(delta) / eta
    m21 = 1j * eta * np.sin(delta)
    m22 = np.cos(delta)
    return np.array([[m11, m12], [m21, m22]], dtype=complex)

def multilayer_transfer_matrix(n_list, d_list, wavelength, n0, theta0, polarization):
    """
    Calculates total transfer matrix for a multilayer stack at oblique incidence.
    """
    M = np.identity(2, dtype=complex)
    for n, d in zip(n_list, d_list):
        # --- MODIFICATION: Pass angle and polarization info to layer_matrix ---
        M = np.matmul(M, layer_matrix(n, d, wavelength, n0, theta0, polarization))
    return M

def transmission(n0, ns, n_list, d_list, wavelengths, theta0_deg, polarization):
    """
    Computes transmission spectrum for oblique incidence.
    theta0_deg: angle of incidence in degrees
    polarization: 'TE' or 'TM'
    """
    T = []
    # --- MODIFICATION: Convert angle to radians once ---
    theta0 = np.deg2rad(theta0_deg)

    # --- MODIFICATION: Calculate effective indices for incident and substrate media ---
    theta_s = np.arcsin((n0 / ns) * np.sin(theta0)) # Angle in substrate
    if polarization.upper() == 'TE':
        eta_0 = n0 * np.cos(theta0)
        eta_s = ns * np.cos(theta_s)
    elif polarization.upper() == 'TM':
        eta_0 = n0 / np.cos(theta0)
        eta_s = ns / np.cos(theta_s)
    else:
        raise ValueError("Polarization must be 'TE' or 'TM'")

    for wl in wavelengths:
        M = multilayer_transfer_matrix(n_list, d_list, wl, n0, theta0, polarization)
        m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
        
        # --- MODIFICATION: Use effective indices in the final calculation ---
        denom = (m11 + m12 * eta_s) * eta_0 + (m21 + m22 * eta_s)
        t = 2 * eta_0 / denom
        T.append(abs(t)**2 * (eta_s / eta_0).real)
    return np.array(T)

# -----------------------------
# Structure definition (no changes here)
# -----------------------------

# Materials (refractive indices)
n_TiO2 = 2.5197
n_SiO2 = 1.4533
n_defect = 1.4626
n_air = 1.0

# Thicknesses (quarter-wave at 800 nm for mirrors)
d_TiO2 = 800 / (4 * n_TiO2)
d_SiO2 = 800 / (4 * n_SiO2)
d_defect = 800 / (2 * n_defect)

# Number of mirror periods on each side
N = 6

# Build stack
n_list = []
d_list = []
# Left Bragg mirror
for _ in range(N):
    n_list.extend([n_TiO2, n_SiO2])
    d_list.extend([d_TiO2, d_SiO2])
# Defect layer
n_list.append(n_defect)
d_list.append(d_defect)
# Right Bragg mirror
for _ in range(N):
    n_list.extend([n_TiO2, n_SiO2])
    d_list.extend([d_TiO2, d_SiO2])

# -----------------------------
# Simulation
# -----------------------------

# --- NEW INPUT PARAMETERS ---
angle_deg = 60  # Angle of incidence in degrees
pol = 'TE'      # Polarization ('TE' or 'TM')

wavelengths = np.linspace(600,1000,1000)  # nm range
# --- MODIFICATION: Call the new transmission function ---
T = transmission(n_air, n_air, n_list, d_list, wavelengths, angle_deg, pol)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(wavelengths, T, 'b')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission")
# --- MODIFICATION: Update title to reflect new parameters ---
plt.title(f"1D Photonic Crystal ({pol} Pol) at {angle_deg}° Incidence")
plt.grid(True)
plt.show()