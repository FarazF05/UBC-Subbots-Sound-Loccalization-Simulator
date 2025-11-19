import numpy as np
import matplotlib.pyplot as plt

from subbots_sim.config import global_vars
from subbots_sim.analysis.math_tools import cylindrical_to_xy


def simulate_continous_signals_for_pinger(pinger_xy, hydrophone_z, pinger_z,
                                num_periods=200, noise_std=0.05):
    """
    Simulate sinusoidal signals received at each hydrophone from a single pinger.

    Physics model (very simple):
      - The pinger emits a continuous sine wave at frequency f0.
      - Sound travels at speed c (speed of sound).
      - Each hydrophone is at a different distance from the pinger,
        so the wave arrives at slightly different times (TOAs).
      - We model each received signal as a time-shifted sine wave + noise.
    """
    # Unpack global variables
    hydrophones_xy = cylindrical_to_xy(global_vars.hydrophone_positions) 
    sampling_rate = global_vars.sampling_frequency # fs
    pinger_freq = global_vars.signal_frequency # f0
    c  = global_vars.speed_of_sound

    # Decide how long to simulate (in samples)
    samples_per_period = int(round(sampling_rate / pinger_freq))

    # Total number of samples we simulate for each hydrophone
    num_samples = num_periods * samples_per_period

    # Time axis for the simulation recording t = 0, 1/f2, 2/f2, ...
    t = np.arange(num_samples) / sampling_rate

    # Compute the distances and time of arrival (TOA) for each hydrophone
    dx = hydrophones_xy[:, 0] - pinger_xy[0]
    dy = hydrophones_xy[:, 1] - pinger_xy[1]
    dz = hydrophone_z - pinger_z

    # Compute straight-line distances
    distances = np.sqrt(dx**2 + dy**2 + dz**2)

    # Compute time of arrivals
    toas = distances / c

    # Build one simluated signal per hydrophone
    signals = []

    for toa in toas:
        # Base idea:
        #   A pure sinusoid is:          sin(2π f0 t)
        #   If it arrives "late" by τ:   sin(2π f0 (t - τ))
        #
        # So we shift the time axis by subtracting the TOA.
        # Each hydrophone just sees a time-shifted version of the same sine wave.
        signal = np.sin(2 * np.pi * pinger_freq * (t - toa))

        # Add Gaussian noise if requested
        if noise_std > 0:
            noise = np.random.normal(scale=noise_std, size=num_samples)
            signal += noise
        
        signals.append(signal)

    return signals


def simulate_pulsed_signals_for_pinger(pinger_xy, hydrophone_z, pinger_z,
                               num_periods=200, noise_std=0.05):
    """
    Simulate time-domain signals at each hydrophone for a *pulsed* pinger.

    The pinger is modeled as:
      - a continuous sine wave at frequency f0 = global_vars.signal_frequency,
      - multiplied by a square-like pulse train with
        frequency pinger_carrier_frequency and duty pinger_duty_cycle,
      - propagating at speed c = global_vars.speed_of_sound in 3D.

    For each hydrophone:
      - we compute its 3D distance to the pinger,
      - convert that distance to a time of arrival TOA_i = distance_i / c,
      - build a local time axis t_local = t - TOA_i,
      - set the signal to (approximately) zero for t_local < 0 (sound has not arrived),
      - and for t_local >= 0, generate a gated sine wave plus optional Gaussian noise.

    Inputs:
        pinger_xy    : array-like of shape (2,), [x_p, y_p] position of the pinger in the XY plane
        hydrophone_z : float, common z-coordinate of all hydrophones
        pinger_z     : float, z-coordinate of the pinger
        num_periods  : int, total number of sine-wave periods of f0 to simulate
        noise_std    : float, standard deviation of additive Gaussian noise

    Returns:
        signals      : list of 1D np.ndarrays, one array per hydrophone,
                       all the same length (num_samples), containing the
                       pulsed, delayed, noisy pinger signals.
    """
     
    # Unpack global variables
    hydrophones_xy = cylindrical_to_xy(global_vars.hydrophone_positions)
    sampling_rate = global_vars.sampling_frequency # fs
    pinger_freq = global_vars.signal_frequency # f0
    c  = global_vars.speed_of_sound

    # Carrier and duty cycle for the pulsed pinger
    carrier_freq = getattr(global_vars, 'pinger_carrier_frequency', None) # Hz
    duty_cycle   = getattr(global_vars, 'pinger_duty_cycle', 1.0) # fraction [0, 1]

    # Decide how long to simulate (in samples)
    duration = num_periods / pinger_freq  # seconds

    # Number of samples to simulate
    num_samples = int(round(duration * sampling_rate))

    # Global time vector for the simulation
    t = np.arange(num_samples) / sampling_rate

    # Compute the distances and time of arrival (TOA) for each hydrophone
    dx = hydrophones_xy[:, 0] - pinger_xy[0]
    dy = hydrophones_xy[:, 1] - pinger_xy[1]
    dz = hydrophone_z - pinger_z
    distances = np.sqrt(dx**2 + dy**2 + dz**2)

    # Compute time of arrivals
    toas = distances / c

    # Build one simluated signal per hydrophone
    signals = []

    # If we have a carrier, precompute its period
    if carrier_freq is not None and carrier_freq > 0:
        carrier_period = 1 / carrier_freq
    else:
        carrier_period = None # Means no pusling just ON 
    
    for toa in toas:
        # Local time axis for this hydrophone:
        #   t_local < 0  → sound hasn't arrived yet (silence)
        #   t_local ≥ 0  → sound is present (gated sine)
        t_local = t - toa

       # Base sine wave at the pinger frequency
       # This will be zerod out before arrival with a gate
        sine_wave = np.sin(2 * np.pi * pinger_freq * t_local)
    
        # Build the gate
        if carrier_period is not None and 0.0 < duty_cycle < 1.0:
            
            # Start with a zero gate
            gate = np.zeros_like(t_local)

            # Only times after arrival can be non-zero
            active_mask = t_local >= 0
            t_after_arrival = t_local[active_mask]

            # Compute the position within the carrier period
            phase_in_period = np.mod(t_after_arrival, carrier_period)

            # ON when in the duty cycle portion
            on_mask = phase_in_period < (duty_cycle * carrier_period)

            # Set gate to 1.0 when both active and ON
            gate[active_mask] = on_mask.astype(float)
        else:
            # Simple gate: 0 before arrival, 1 after arrival
            gate = (t_local >= 0).astype(float)

        # Final signal at the hydrophone
        # 0 before sound arrives
        # pulsed sine after arrival
        signal = gate * sine_wave

        # Add Gaussian noise if requested
        if noise_std > 0:
            noise = np.random.normal(scale=noise_std, size=num_samples)
            signal += noise
        
        signals.append(signal)
    
    return signals


def plot_hydrophone_signal(signals, hydrophone_index=0, title_prefix="Hydrophone"):
    """
    Non-blocking plot of one hydrophone signal.
    """
    fs = global_vars.sampling_frequency

    sig = np.asarray(signals[hydrophone_index])
    num_samples = sig.size
    t = np.arange(num_samples) / fs

    plt.figure()                      # new figure, separate from animation
    plt.plot(t, sig)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"{title_prefix} {hydrophone_index} signal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)             # non-blocking
    plt.pause(0.001)                  # give GUI a moment to draw