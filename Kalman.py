"""This script contains both the total Kalman filter as the Kalman filter per sub-channel.
The filters are based on the following publication:
KALMAN-FILTER CHANNEL ESTIMATOR FOR OFDM SYSTEMS IN TIME AND
FREQUENCY-SELECTIVE FADING ENVIRONMENT Wei Chen and Ruifeng Zhang (2004)"""

# Imports
import numpy as np
from scipy.special import j0
import matplotlib.pyplot as plt
from scipy import signal

def One_Kalman_filter(pilot_symbols, variance_w, data_symbols, pilot, FFT_length, Channel_length, fd, sampling_freq):
    """This function defines the total Kalman filter based on the mentioned paper above.
    Input:
        - pilot_symbols: The received pilot symbols
        - variance_w: The variance of the corresponding noise level of the received symbols
        - data_symbols: The received data symbols
        - pilot: The true pilot symbols
        - FFT_length: The defined FFT length
        - Channel_length: The defined channel length
        - fd: The Doppler frequency
        - sampling_freq: The defined sampling frequency
    Output:
        - hat_hn: The channel estimation """
    # Initiate the max delay spread
    sigma_t = 25
    # Calculate the sample duration
    T = (FFT_length + (Channel_length -1))/sampling_freq

    # Initiate empty parameters for the Kalman filter
    Sigma_prev = np.zeros((200, 200), dtype=np.complex128)
    Shifted = np.zeros((200, 200), dtype=np.complex128)
    G = np.identity(200, dtype=np.complex128)
    D = np.diag(np.full(200, pilot))
    x_prev = np.zeros((200,), dtype=np.complex128)
    M_n = np.zeros((7,200), dtype=np.complex128)
    hat_hn = np.zeros((7, 200), dtype=np.complex128)
    reshaped_data_symbols = np.zeros((7, 200, 200), dtype=np.complex128)

    # Iterate over 0 and 1
    for m in range(2):
        # Calculate the defined correlation coefficient
        coefficient = j0(2*np.pi*fd * m * T)
        # Iterate over the amount of pilot symbols
        for k in range(200):
            for l in range(200):
                # Assign the corresponding coefficient
                if m == 1:
                    Sigma_prev[k,l] = coefficient
                elif m == 0:
                    Shifted[k,l] = coefficient
    # Initiate the transition matrix
    A = np.dot(np.dot(Shifted**2, abs(Sigma_prev)**2), np.linalg.pinv(Sigma_prev))
    C = -A
    # Iterate over the index and pilot symbol of the received pilot symbols
    for t, symbols_at_T in enumerate(pilot_symbols):
        # Calculate the Kalman iteration as defined in the mentioned paper
        M_n = np.dot(np.dot(C, Sigma_prev), C.conjugate()) + np.dot(G, G.conjugate())
        lambda_ = np.dot(np.dot(D, M_n), D.conjugate()) + np.dot(variance_w**2, np.identity(200))
        # Calculate the Kalman gain
        K = np.dot(np.dot(M_n, D.conjugate()), np.linalg.inv(lambda_))
        # Calculate the updated state
        xn = np.dot(C, x_prev) + np.dot(K, (symbols_at_T - np.dot(np.dot(D, C), x_prev)))
        # Calculate the updated Sigma matrix of the corresponding new state
        Sigma_n = np.dot((np.identity(200) - np.dot(K, D)), M_n)
        # Set the new state and new sigma as the previous state and sigma
        x_prev = xn
        Sigma_prev = Sigma_n
        # Create the channel estimation
        hat_hn[t] = np.dot(np.identity(200, dtype=np.complex128), xn) # Same as xn
    # Return the channel estimation
    return hat_hn


def Kalman_filter_per_channel(pilot_symbols, variance_w, data_symbols, pilot, FFT_Length, Channel_length, fd, sampling_freq):
    """This function defines the per sub-channel Kalman filter based on the mentioned paper above.
    Input:
        - pilot_symbols: The received pilot symbols
        - variance_w: The variance of the corresponding noise level of the received symbols
        - data_symbols: The received data symbols
        - pilot: The true pilot symbols
        - FFT_length: The defined FFT length
        - Channel_length: The defined channel length
        - fd: The Doppler frequency
        - sampling_freq: The defined sampling frequency
    Output:
        - xn: The channel estimation """
    # Calculate the sample duration
    T = (FFT_Length+(Channel_length - 1))/sampling_freq
    # Initiate the parameters for the Kalman filter
    Sigma_n = np.zeros((7,200), dtype=np.complex128)
    M_n = np.zeros((7,200), dtype=np.complex128)
    auto_correlation = [j0(2*np.pi*fd * 0 * T), j0(2*np.pi*fd * 1 *T)] # By equation from paper
    Sigma_prev = np.full((200),(auto_correlation[0]**2 - abs(auto_correlation[1])**2) / auto_correlation[0])
    lambda_ = np.zeros((7,200), dtype=np.complex128)
    K = np.zeros((7,200), dtype=np.complex128)
    xn = np.zeros((7,200), dtype=np.complex128)
    a = auto_correlation[1] / auto_correlation[0]
    a = -a
    C = a
    xn_prev = np.full((200) ,0)
    g = 1

    # Iterate over the received pilot symbols
    for t, symbols_at_T in enumerate(pilot_symbols):
        # Calculate the Kalman iteration as defined in the mentioned paper
        M_n[t,:] = C * Sigma_prev * np.conjugate(C) + 1
        if M_n[t,:].shape != (200,):
            raise Exception(f"M_N at T: {M_n[t,:].shape}")

        lambda_[t,:] = pilot * M_n[t,:] * np.conjugate(pilot) + variance_w
        # Calculate the Kalman gain
        K[t,:] = M_n[t,:] * np.conjugate(pilot) / lambda_[t,:]

        if symbols_at_T.shape != (200,):
            raise Exception(f"Symbols at T: {symbols_at_T.shape}")
        # Calculate the new state
        xn[t,:] = C*xn_prev + K[t,:] * (symbols_at_T - pilot*C* xn_prev)
        # Calculate the new sigma
        Sigma_n[t,:] = (1- K[t,:] * pilot) * M_n[t,:]
        # Set the new sigma and state as the previous state and sigma
        xn_prev = xn[t,:]
        Sigma_prev = Sigma_n[t,:]
    # Return the channel estimation
    return xn


def linear_combiner(hn):
    """This function applies a lowpass filter to the subchannel estimates to filter out high frequency noise in the estimates  
    Input:
        - hn: An np array of the channel estimations per subchannel
    Output:
        - hn_filtered: An np array of the low pass filtered channel estimations per subchannel """

    # Create the lowpass filter. Parameters found to be optimal through trial and error
    cutoff = 0.98
    window_size = 11
    lowpass = signal.firwin(window_size, cutoff)

    # Pad the hn array with the first and last values to avoid edge effects
    hn_pad = np.concatenate((np.tile(hn[:, :1], (1, window_size//2)), hn, np.tile(hn[:, -1:], (1, window_size//2))), axis=1)
    print("shape hn: ", hn_pad.shape)
    # Apply the lowpass filter to the hn array
    hn_filtered = np.apply_along_axis(lambda x: np.convolve(x, lowpass, mode='valid'), axis=-1, arr=hn_pad)

    print(hn_filtered.shape)
    return hn_filtered