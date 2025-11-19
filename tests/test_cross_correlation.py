import numpy as np

from subbots_sim.analysis.cross_correlation import calc_cross_correlation

def test_calc_cross_correlation_known_phase():
    fs = 100_000      # sampling frequency (Hz)
    f = 1_000         # signal frequency (Hz)
    t = np.arange(0, 1, 1/fs)

    # True phase shift in radians
    phase_shift = np.pi / 2   # 90 degrees

    sig1 = np.sin(2 * np.pi * f * t)
    sig2 = np.sin(2 * np.pi * f * t + phase_shift)

    # Our function returns phase difference in degrees
    phase_diff_deg = calc_cross_correlation((sig1, sig2), fs, f)

    # Convert true phase shift to degrees
    true_phase_deg = phase_shift * 180 / np.pi

    assert np.isclose(phase_diff_deg, true_phase_deg, atol=2.0)
