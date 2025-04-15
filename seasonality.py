import numpy as np


def generate_fourier_series(t, num_harmonics, period):
    """
    Generate Fourier series features up to num_harmonics for time points t.

    Parameters:
    -----------
    t : array-like
        Time points.
    num_harmonics : int
        Number of harmonic pairs.
    period : float
        Seasonal period.

    Returns:
    --------
    A NumPy array of shape (len(t), 2*num_harmonics) containing the sine and cosine features.
    """
    t = np.array(t, dtype=float)
    features = []
    for t_val in t:
        row = []
        for k in range(1, num_harmonics + 1):
            row.append(np.sin(2 * np.pi * k * t_val / period))
            row.append(np.cos(2 * np.pi * k * t_val / period))
        features.append(row)
    return np.array(features)
