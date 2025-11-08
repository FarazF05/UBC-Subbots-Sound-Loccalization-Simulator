import global_vars
import numpy as np
from scipy.signal import correlate

# INPUT: n sine waves or csv data
# OUTPUT: n-1 phase difference
def cross_correlation_stage(signal):
    # number of components
    num_components = len(global_vars.hydrophone_positions) - 1

    phase_analysis_inputs = [
            (signal[0], signal[i+1])
            for i in range(num_components)
    ]

    phase_differences = [
        calc_cross_correlation(pair)
        for pair in phase_analysis_inputs
    ]

    return phase_differences

# INPUT: input signal tuple
# OUTPUT: ideally just returns phase difference 
def calc_cross_correlation(signal_pair):
    # compute cross correlation
    h0_sig, h_sig = signal_pair
    cross_correlation = correlate(h0_sig, h_sig, mode='same')
    
    # find discrete signal frequency
    f = global_vars.signal_frequency / global_vars.sampling_frequency
    
    # find number of samples per signal period
    N = int(np.ceil(1/f))
    
    # constrain region of interest based on signal periodicity
    # note that the correlation distribution is centered at n=0
    center_idx = int(len(cross_correlation)/2)
    halfrange = int(N/2)
    roi = cross_correlation[center_idx-halfrange:center_idx+halfrange]

    # index offset (where cross correlation peaks)
    maxima_idx = np.argmax(roi) - halfrange
    time_delay = maxima_idx / global_vars.sampling_frequency
    
    # calculate the phase difference tdoa
    phase_difference = 2 * np.pi * global_vars.signal_frequency * time_delay

    return phase_difference;
