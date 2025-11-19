import numpy as np
from scipy.signal import correlate, resample_poly

from subbots_sim.config import global_vars

def cross_correlation_stage(signals):
    """
    Given a list of hydrophone signals, compute the TDOAs Δt_i = ti - t0
    between hydrophone 0 and each other hydrophone using cross-correlation.

    Inputs:
        signals: list of length N of 1D np.ndarrays

    Returns:
        tdoas: list of length N-1 of floats (seconds)
               element i-1 corresponds to hydrophone i relative to 0
    """
    num_hydrophones = len(signals)
    ref_signal = signals[0]

    # Stores one TDOA per hydrophone (relative to hydrophone 0)
    tdoas = []

    for i in range(1, num_hydrophones):
        other_signal = signals[i]

        # Use cross-correlation to estimate time delay between ref and other hydrophone
        delay_seconds = calc_cross_correlation(
            (ref_signal, other_signal),
            global_vars.sampling_frequency,
            global_vars.signal_frequency,
        )

        tdoas.append(delay_seconds)
    return tdoas



def calc_cross_correlation(signal_pair, sampling_frequency, signal_frequency):
    """
    Estimate the time delay between two hydrophone signals using cross-correlation.

    The first signal (h0_sig) is treated as the reference.
    The second signal (h_sig) is the one we compare to it.

    Inputs:
        signal_pair: tuple (h0_sig, h_sig)
            h0_sig and h_sig are 1D numpy arrays with the same length.
        sampling_frequency: float
            Sampling rate in Hz.
        signal_frequency: float
            Pinger frequency in Hz.

    Returns:
        time_delay: float
            Estimated delay in seconds.
            Positive means h_sig arrives LATER than h0_sig.
    """

    # Unpack the two signals
    h0_sig, h_sig = signal_pair

    # Figure out how many samples are in one peroid of the signal 
    samples_per_period = int(np.ceil(sampling_frequency / signal_frequency))

    # Create a window size of 100 periods
    window_size = 100 * samples_per_period

    # Take equal-length windows around the centers of both signals
    center0 = len(h0_sig) // 2
    center1 = len(h_sig) // 2

    strart0 = center0 - window_size // 2
    end0   = center0 + window_size // 2

    start1 = center1 - window_size // 2
    end1   = center1 + window_size // 2

    seg0 = h0_sig[strart0:end0]
    seg1 = h_sig[start1:end1]

    # Make sure the two segments are the same length
    if seg0.shape[0] != seg1.shape[0]:
        raise Exception("Error: Extracted segments are not the same e=length")
    
    # Compute cross-correlation over all possible lags
    # Correlate(seg0, seg1, mode='full) gives lag from (len(seg1)-1) to (len(seg0)-1)
    corr = correlate(seg0, seg1, mode='full')

    # Build the corresponding lag values in samples
    num_samples = seg0.size
    lags = np.arange(-num_samples + 1, num_samples)

    # Keep only lags within ± one period of the signal
    max_lag_samples = samples_per_period
    valid_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)

    corr_region = corr[valid_mask]
    lags_region = lags[valid_mask]

    # Find the lag that gives the maximum correlation in this region
    best_index = np.argmax(corr_region)
    best_lag_samples = lags_region[best_index]  # can be negative or positive

    # Convert lag (in samples) to TDOA Δt = t0 - ti
    time_delay = best_lag_samples / sampling_frequency

    return time_delay


def compute_measured_tdoas_from_signals(signals):
    """
    Compute measured TDOAs from hydrophone signals.

    This function runs the cross-correlation stage on the list of hydrophone
    signals to obtain time delays between hydrophone 0 and each other
    hydrophone.

    Inputs:
        signals: list (size N) of arrays: hydrophone data

    Returns:
        measured_tdoas: (N-1,) array of measured TDOAs (seconds),
            ordered as element i-1 corresponds to hydrophone i relative to 0.
            measured_tdoas[i-1] is the TDOA Δt_i = t_i - t_0 (seconds)
    """
    
    # Use cross-correlation to compute TDOAs
    tdoas_list = cross_correlation_stage(signals)

    # Convery the list to a numpy array
    measured_tdoas = np.asarray(tdoas_list, dtype=float)

    return measured_tdoas
