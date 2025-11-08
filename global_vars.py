from common_types import *

hydrophone_positions = [
    CylindricalPosition(0, 0, 0),
    CylindricalPosition(1.85e-2, 0, -1e-2),
    CylindricalPosition(1.85e-2, np.pi/2, -1e-2),
    CylindricalPosition(1.85e-2, np.pi, -1e-2),
    CylindricalPosition(1.85e-2, -np.pi/2, -1e-2),
]

signal_frequency = 40e3
sampling_frequency = 20*signal_frequency


