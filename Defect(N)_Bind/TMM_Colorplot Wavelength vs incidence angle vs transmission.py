import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Transfer Matrix Method (TMM) for OBLIQUE INCIDENCE
# ------------------------------------------------------------------

def layer_matrix_oblique(n, d, wavelength, theta0, n0, polarization):
    """
    Returns transfer matrix for a single layer at oblique incidence.
    """
    # Calculate angle inside the layer using Snell's Law
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
# Structure Definition
# ------------------------------------------------------------------
n_TiO2 = 2.5197
n_SiO2 = 1.4533
n_defect = 1.462605
n_air = 1.0

# Thicknesses designed for a central wavelength of 800 nm
d_TiO2 = 800 / (4 * n_TiO2)  # Quarter-wave thickness
d_SiO2 = 800 / (4 * n_SiO2)  # Quarter-wave thickness
d_defect = 800 / (2 * n_defect) # Half-wave thickness

# Number of mirror periods on each side of the defect
N = 6

# Build the layer stack
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
# Simulation: Store full spectra for each angle
# ------------------------------------------------------------------
wavelengths = np.linspace(400, 1400, 1000)  # Wavelength range in nm
angles = np.linspace(0, 80, 50)           # Angle range in degrees

# Initialize 2D arrays to store the colormap data.
# Shape will be (num_wavelengths, num_angles)
transmittance_map_te = np.zeros((len(wavelengths), len(angles)))
transmittance_map_tm = np.zeros((len(wavelengths), len(angles)))

print("Calculating...")
# Use enumerate to get the index for storing data in the arrays
for i, angle in enumerate(angles):
    # Calculate the full transmission spectrum for the current angle
    T_te = transmission_oblique(n_air, n_air, n_list, d_list, wavelengths, angle, 'TE')
    T_tm = transmission_oblique(n_air, n_air, n_list, d_list, wavelengths, angle, 'TM')

    # Store the calculated spectrum as a column in our 2D array
    transmittance_map_te[:, i] = T_te
    transmittance_map_tm[:, i] = T_tm

print("Calculation complete.")

# ------------------------------------------------------------------
# Plotting: Generate Colormaps with Wavelength on X-axis
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.suptitle('Angular Dispersion of Transmittance', fontsize=16)
vmax = 1.0 # Use a common color scale for both plots

# --- TE Polarization Colormap ---
# X-axis is wavelengths, Y-axis is angles.
# The data array must be transposed with .T to match the axes.
im1 = ax1.pcolormesh(wavelengths, angles, transmittance_map_te.T,
                     shading='gouraud', cmap='viridis', vmax=vmax)
ax1.set_title('TE Polarization')
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Incidence Angle (degrees)")
fig.colorbar(im1, ax=ax1, label='Transmittance')

# --- TM Polarization Colormap ---
im2 = ax2.pcolormesh(wavelengths, angles, transmittance_map_tm.T,
                     shading='gouraud', cmap='viridis', vmax=vmax)
ax2.set_title('TM Polarization')
ax2.set_xlabel("Wavelength (nm)")
# No y-label needed here as it's shared with the first plot
fig.colorbar(im2, ax=ax2, label='Transmittance')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()