import numpy as np
from collections import namedtuple

from subbots_sim.common_types import *

hydrophone_positions = [
    # Positions of 4 hydrophones in cylindrical coordinates (r, theta, z), meters
    CylindricalPosition(0, 0, 0),
    CylindricalPosition(1.85e-2, 0, -1e-2),
    CylindricalPosition(1.85e-2, np.pi/2, -1e-2),
    CylindricalPosition(1.85e-2, np.pi, -1e-2),
    CylindricalPosition(1.85e-2, -np.pi/2, -1e-2),
]

signal_frequency = 40e3 # Hz
sampling_frequency = 20*signal_frequency # Hz

speed_of_sound = 1482  # m/s in water at ~20 degrees C

# How the pinger is pulsed
pinger_carrier_frequency = 50.0   # Hz
pinger_duty_cycle        = 0.8   # # of each carrier period is "ON"

# Shits itself with 200 Hz and 10% duty cycle praying the comp is more forgiving