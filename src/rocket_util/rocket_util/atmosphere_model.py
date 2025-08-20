import numpy as np

def atmosphere_profile(alti):
    """
    Obtain real (unscaled) atmosphere profile by altitude.
    Based on NASA's interpolation for temperature and pressure with altitude.

    Parameters:
        h (array-like): Altitude(s) in meters.

    Returns:
        dict: Profile with temperature (K), pressure (Pa), density (kg/m^3), and altitude (m).
    """
    alti = np.asarray(alti)
    h_trop = alti[alti < 10999]
    h_lstrat = alti[(alti >= 10999) & (alti < 24999)]
    h_ustrat = alti[alti >= 24999]

    # Temperature profiles (Celsius)
    T_trop = 15.04 - 0.00649 * h_trop
    T_lstrat = -56.46 * np.ones_like(h_lstrat)
    T_ustrat = -131.21 + 0.00299 * h_ustrat

    # Pressure profiles (kPa)
    P_trop = 101.29 * ((T_trop + 273.15) / 288.08) ** 5.256
    P_lstrat = 22.65 * np.exp(1.73 - 0.000157 * h_lstrat)
    P_ustrat = 2.488 * ((T_ustrat + 273.15) / 216.6) ** -11.388

    # Concatenate results
    T = np.concatenate([T_trop, T_lstrat, T_ustrat]) + 273.15  # Kelvin
    P = np.concatenate([P_trop, P_lstrat, P_ustrat]) * 1000    # Pascal
    rho = P / (286.9 * T)

    # Reorder to match input h
    h_all = np.concatenate([h_trop, h_lstrat, h_ustrat])
    order = np.argsort(np.argsort(alti))
    T = T[order]
    P = P[order]
    rho = rho[order]

    return {
        "T": T,
        "P": P,
        "rho": rho,
    }
