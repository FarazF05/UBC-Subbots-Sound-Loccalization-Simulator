import numpy as np

def cylindrical_to_xy(cyl_positions):
    """
    Convert list of CylindricalPosition(r, theta, z) to 2D Cartesian (x, y).

    We ignore z for the top-down plot.
    """
    xy = []
    for c in cyl_positions:
        r, theta, z = c
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        xy.append((x, y))
    return np.array(xy)