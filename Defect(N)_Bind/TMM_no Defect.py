import numpy as np
import matplotlib.pyplot as plt

# Code simulates a 1D photonic crytal without a defect layer
# -----------------------------
# Transfer Matrix Method (TMM)
# -----------------------------

def layer_matrix(n, d, wavelength):
    """
    Returns transfer matrix for a single dielectric layer.
    n: refractive index of the layer
    d: thickness of the layer (in nm)
    wavelength: free-space wavelength (in nm)
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
    Calculates total transfer matrix for a multilayer stack.
    n_list: list of refractive indices [n1, n2, ...]
    d_list: list of thicknesses [d1, d2, ...] (same length as n_list)
    wavelength: free-space wavelength (nm)
    """
    M = np.identity(2, dtype=complex)
    for n, d in zip(n_list, d_list):
        M = np.matmul(M, layer_matrix(n, d, wavelength))
    return M

def transmission(n0, ns, n_list, d_list, wavelengths):
    """
    Computes transmission spectrum.
    n0: refractive index of incident medium (air = 1)
    ns: refractive index of substrate (air = 1)
    n_list, d_list: layer refractive indices and thicknesses
    wavelengths: array of wavelengths (nm)
    """
    T = []
    for wl in wavelengths:
        M = multilayer_transfer_matrix(n_list, d_list, wl)
        m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
        denom = (m11 + m12*ns) * n0 + (m21 + m22*ns)
        t = 2 * n0 / denom
        T.append(abs(t)**2 * (ns/n0).real)
    return np.array(T)

# -----------------------------
# Structure definition
# -----------------------------

# Materials (refractive indices)
n_TiO2 = 2.5197
n_SiO2 = 1.4533
n_defect = 1.4634
n_air = 1.0

# Thicknesses (quarter-wave at 800 nm for mirrors)
d_TiO2 = 800 / (4 * n_TiO2)  # ≈ 79.4 nm
d_SiO2 = 800 / (4 * n_SiO2)  # ≈ 137.6 nm
d_defect = 800 / (2 * n_defect)  # ≈ 273.3 nm

# Number of mirror periods on each side
N = 6  # you can adjust this

# Build stack: (Air) – (N periods of [TiO2, SiO2]) – Defect – (N periods of [TiO2, SiO2]) – (Air)
n_list = []
d_list = []

# Left Bragg mirror
for _ in range(N):
    n_list.extend([n_TiO2, n_SiO2])
    d_list.extend([d_TiO2, d_SiO2])

# Right Bragg mirror
for _ in range(N):
    n_list.extend([n_TiO2, n_SiO2])
    d_list.extend([d_TiO2, d_SiO2])

# -----------------------------
# Simulation
# -----------------------------
wavelengths = np.linspace(600, 1200, 1000)  # nm range
T = transmission(n_air, n_air, n_list, d_list, wavelengths)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(wavelengths, T, 'b', linewidth=2)
plt.xlabel("Wavelength (nm)", fontsize=20)
plt.ylabel("Transmission", fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1.05)
plt.show()

