import numpy as np
import matplotlib.pyplot as plt
# Code shows the binding of Atrazine to MIP by simulating a change in refractive index of the defect layer

# -----------------------------
# Transfer Matrix Functions
# -----------------------------

def layer_matrix(n, d, wavelength):
    """Transfer matrix for one layer at normal incidence"""
    k0 = 2 * np.pi / wavelength
    delta = k0 * n * d
    m11 = np.cos(delta)
    m12 = 1j * np.sin(delta)/n
    m21 = 1j * n * np.sin(delta)
    m22 = np.cos(delta)
    return np.array([[m11, m12],[m21,m22]], dtype=complex)

def multilayer_matrix(n_list, d_list, wavelength):
    """Total transfer matrix of the stack"""
    M = np.identity(2, dtype=complex)
    for n,d in zip(n_list,d_list):
        M = M @ layer_matrix(n,d,wavelength)
    return M

def transmission(n0, ns, n_list, d_list, wavelengths):
    """Compute transmission spectrum"""
    T = []
    for wl in wavelengths:
        M = multilayer_matrix(n_list, d_list, wl)
        m11,m12 = M[0,0], M[0,1]
        m21,m22 = M[1,0], M[1,1]
        denom = (m11 + m12*ns)*n0 + (m21 + m22*ns)
        t = 2*n0 / denom
        T.append(abs(t)**2 * (ns/n0).real)
    return np.array(T)

# -----------------------------
# Structure parameters
# -----------------------------
# Materials
n_TiO2 = 2.5197
n_SiO2 = 1.4533
n_air = 1.0

# Thicknesses (nm)
d_TiO2 = 800/(4*n_TiO2)  # 83.3 nm
d_SiO2 = 800/(4*n_SiO2)  # 142.9 nm
d_defect = (800/(2*1.46))   # 267 nm

# Number of periods on each side
N = 8

# Wavelength range (nm)
wavelengths = np.linspace(900, 1000, 1000)

# Base stack without defect
n_base = []
d_base = []

for _ in range(N):
    n_base.extend([n_TiO2,n_SiO2])
    d_base.extend([d_TiO2,d_SiO2])

# -----------------------------
# Simulate defect refractive index variation
# -----------------------------
defect_indices = np.array([1.3961,1.4452,1.4880,1.4982,1.5078,1.5091,1.5102,1.5104])  # Example: small changes to simulate binding

plt.figure(figsize=(7,5))

for n_defect in defect_indices:
    # Complete stack with defect
    n_stack = n_base + [n_defect] + n_base
    d_stack = d_base + [d_defect] + d_base
    
    T = transmission(n_air, n_air, n_stack, d_stack, wavelengths)
    plt.plot(wavelengths, T, label=f'n_defect={n_defect}')

plt.xlabel("Wavelength (nm)", fontsize=20)
plt.ylabel("Transmission", fontsize=20)
plt.legend()
plt.grid(True)
plt.show()
