"""This script contains all the functions to preprocess the received data stream
from the given mat files.
It removes the cyclic prefix from the sequence and performs DFT to go to
the frequency domain. Finally, it extracts the pilot symbols
and data symbols from the frequency domain."""

#Imports
import numpy as np
import matplotlib.pyplot as plt


def remove_cyclic_prefix(signal, length, FFT_length):
    """This function removes the cyclic prefix from the received sequence and returns
    the frames without the cyclic prefix in the time domain.
    Input:
        - signal: The received signal
        - length: The length of the cyclic prefix
        - FFT_length: The length of the FFT
    Output:
        - sub_signal: The extracted frames from the received signal in the time domain"""
    # Initiate an empty list to save the resulting sequences after cyclic prefix removal
    sub_signal = []
    # Iterate over the amount the start indices of the cyclic prefix
    for prefix in range(0, len(signal), length+FFT_length):
        # Set the start index of the cyclic prefix
        start_cyclic_prefix = prefix
        # Set the end index of the cyclic prefix by adding the length of the defined prefix
        end_cyclic_prefix = prefix + length
        # Extract the frame without the cyclic prefix of the same lenght as the defined FFT length
        sub = signal[end_cyclic_prefix:end_cyclic_prefix+FFT_length]
        # Sanity check the length of the extracted frame
        if len(sub) == FFT_length:
            # If correct, save the extracted frame to the list
            sub_signal.append(sub)
    # Return the list as a numpy array
    return np.array(sub_signal)

def to_frequency_domain_subsequences(sequence, Channel_length, FFT_Length):
    """This function extracts the cyclic prefix from the received input
    signal. Next, it performs DFT to get the frames to the freqeuncy domain.
    Input:
        - sequence: The received signal
        - Channel_length: The length of the channel
        - FFT_Length: The length of the FFT
    Output:
        - freq_list: Numpy array with the frames in the frequency domain"""
    # Remove the cyclic prefix
    sub_sequences = remove_cyclic_prefix(sequence, Channel_length - 2, FFT_Length)
    # Initiate an empty list to save the frequency domain frames to
    freq_list = []
    # Iterate over the frames
    for sub in sub_sequences:
        # Make sure, the correct shape is maintained
        sub = np.reshape(sub, (FFT_Length,))
        # Perform DFT (Numpy defines this as FFT, however this is the DFT)
        freq_sub = np.fft.fft(sub)
        # Save the frequency domain frame to the list
        freq_list.append(freq_sub)
    # Return the list as a numpy array
    return np.array(freq_list)

def extract_pilot_symbols(frequency_domain_signals, matfile):
    """This function extract the defined pilot symbols from the given pilot indices.
    Input:
        - frequency_domain_signals: The frames in the frequency domain
        - matfile: The corresponding loaded mat file
    Output:
        - pilots: The extracted pilot symbols
        - pilot_indices: The corresponding pilot indices """
    # Extract the indices of the pilot symbols for python
    pilot_indices = matfile['OFDM']['PilotIndices']-1
    # Initiate an empty list
    pilots = []
    # Iterate over the frames in the frequency domain
    for sample in frequency_domain_signals:
        # Extract the defined pilot symbols from the selected frame
        pilot_symbols = sample[pilot_indices[0][0][0]]
        # Sanity check the shape of the extracted symbols and append this to the list
        pilots.append(np.reshape(np.array(pilot_symbols), (200,)))
    # Return the extracted pilot symbols and corresponding pilot indices
    return np.array(pilots), pilot_indices

def extract_data_symbols(frequency_domain_signals, matfile):
    """This function extract the defined data symbols from the given data indices.
    Input:
        - frequency_domain_signals: The frames in the frequency domain
        - matfile: The corresponding loaded mat file
    Output:
        - pilots: The extracted data symbols
        - pilot_indices: The corresponding data indices """
    # Extract the indices of the data symbols for python
    data_indices = matfile['OFDM']['DataIndices']-1
    # Initiate an empty list
    data = []
    # Iterate over the frames in the frequency domain
    for sample in frequency_domain_signals:
        # Extract the defined data symbols from the selected frame
        data_symbols = sample[data_indices[0][0][0]]
        # Sanity check the shape of the extracted symbols and append this to the list
        data.append(np.reshape(np.array(data_symbols), (1000,)))
    # Return the extracted data symbols and corresponding data indices
    return np.array(data), data_indices
