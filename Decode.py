"""This script contains the functions to reconstruct the image given the filtered
received data symbols. It extracts the symbols by applying the Kalman filters.
Afterwards, the zerod-padding is removed. From the remaining signal, the phase
is extracted and the corresponding bits are assigned based on the QPSK constellation."""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Kalman import One_Kalman_filter, Kalman_filter_per_channel
from OpenData import to_frequency_domain_subsequences, extract_pilot_symbols, extract_data_symbols

def ofdm_equalizer(hn, pilot_symbols, data_symbols):
    """This function perfoms zero forcing equalization by dividing the data symbols
    by the generated channel estimator using element wise division.
    Input:
        - hn: The estimated channel
        - pilot_symbols: The extracted pilot symbols
        - data_symbols: The extracted data symbols.
    Output:
        - retrieved_data: The retrieved data after channel estimation
        - zn_pilot: The retrieved pilots after channel estimation"""
    # Element wise division of the pilot symbols by the estimated channel
    zn_pilot = np.divide(pilot_symbols, hn)
    # Flatten the pilot symbols
    zn_pilot = flatten_pilot_symbols(zn_pilot)
    # Repeat the estimated channel 5 times to be able to extract the data symbols
    hn_data = np.repeat(hn, 5, axis=1)
    # Element wise division of the data symbols by the repeated channel estimator
    retrieved_data = np.divide(data_symbols, hn_data)
    # Return the retrieved data and pilot indices after channel estimation
    return retrieved_data, zn_pilot


def remove_zero_padding(retrieved_data_symbols, image_shape):
    """This function removes the zero padding from the data symbols.
    Input:
        - retrieved_data_symbols: The retrieved data symbols after applying the estimated
        channel
        - image_shape: The given image shape
    Output:
        - no_padding_seq: The data symbols without the zero padding"""
    # Define the desired sequency length given the image shapes
    seq_length = image_shape[0] * image_shape[1]
    # Initiate an empty list
    total_seq = []
    # Iterate over the retrieved data symbols
    for sub_sequence in retrieved_data_symbols:
        # Extend the initiated list wit the retrieved data symbols
        total_seq.extend(sub_sequence)
    # Multiply the desired sequency lenght by 2 to get the amount of bits
    total_length = 2* len(total_seq)
    # Extract the amount of zero paddings by calculating the difference between the
    # desired length and received data length
    zero_padding_len = int((total_length - seq_length)/2)
    # Remove the zero padding
    no_padding_seq = np.array(total_seq)[:-zero_padding_len]
    # Return the data symbols without the zero padding
    return no_padding_seq

def flatten_pilot_symbols(retrieved_pilot_sequence):
    """This function flattens the pilot symbols given the 2D input of the retrieved
    pilot symbols.
    Input:
        - retrieved_pilot_symbols: The extracted pilot symbols after channel estimation
    Output:
        - total_seq: The flattened numpy array"""
    # Initiate an empty list
    total_seq = []
    # Iterate over the retrieved pilot symbols
    for sub_sequence in retrieved_pilot_sequence:
        # Extend the list with the selected pilot symbols
        total_seq.extend(sub_sequence)
    # Return the list as a numpy array
    return np.array(total_seq)

def extract_phase(retrieved_sequence):
    """Extrac the corresponding phase of the retrieved data sequence after channel
    estimation
    Input:
        - retrieved_sequence: the extracted channel estimated data symbols
    Output:
        - degrees: The retrieved radians of the phases"""
    # Set an empty array
    degrees=np.zeros((retrieved_sequence.shape))
    # Iterate over the retrieved data symbols
    for t, symbol in enumerate(retrieved_sequence):
        # Convert the complex values to radians
        degrees[t] = np.angle(symbol)
    # Return the numpy array of radians
    return degrees

def assign_bits(retrieved_sequence):
    """Assign the bits of the extracted phases in radians. The bits are assigned
    based on the QPSK constellation.
    Input:
        - retrieved_sequence: Extracted data symbols after channel estimation
    Output:
        - bit_seq: The extracted bit sequence """
    # Extract the phase
    degrees = extract_phase(retrieved_sequence)
    # Initiate an empty list
    bit_seq = []
    # Iterate over the index and the extracted radians
    for t, k in enumerate(degrees):
        # Define the ranges with the corresponding bits of the constellation
        if k >= 0 and k < (np.pi/2):
            bits = [0,0]
        elif k >= (np.pi/2) and k < np.pi:
            bits = [0, 1]
        elif k > -(np.pi) and k <= -(np.pi/2) or k == np.pi:
            bits = [1, 1]
        else:
            bits = [1, 0]
        # Append the extracted bits to the list
        bit_seq.append(np.array(bits))
    # Return the list of bits as a numpy array
    return np.array(bit_seq)

def extract_received_signal(signal, variance_w, pilot, Channel_length, FFT_length, matfile):
    """This function combines all the steps to reconstruct the image based on the
    loaded sequency of the received signal.
    Input:
        - signal: The loaded received signal
        - variance_w: The defined variance of the corresponding noise level of the received signal
        - pilot: The given pilot symbol
        - Channel_length: The given channel length
        - FFT_length: The given FFT length
        - matfile: The loaded matfile
    Output:
        - bits_pilot: The extracted bits of the pilot symbols generated by the total Kalman filter
        - bits_data: The extracted bits of the data symbols generated by the total Kalman filter
        - bits_pilot_single: The extracted bits of the pilot symbols generated by the per-subchannel Kalman filter
        - bits_data_single: The extracted bits of the data symbols generated by the per-subchannel Kalman filter """
    # Convert the signal to the frequency domain
    frequency_domain_signals = to_frequency_domain_subsequences(signal, Channel_length, FFT_length)
    # Extract the pilot symbols and data symbols
    pilot_symbols, pilot_indices = extract_pilot_symbols(frequency_domain_signals, matfile)
    data_symbols, data_indices = extract_data_symbols(frequency_domain_signals, matfile)

    # Extract the doppler frequency and the sampling frequency
    fd = matfile['Channel']['DopplerFreq'][0][0][0]
    sampling_freq = matfile['OFDM']['SamplingFreq'][0][0][0]

    # Generate the channel estimators using either the Kalman filter per sub-channel or the total Kalman filter
    h_n_single = Kalman_filter_per_channel(pilot_symbols, variance_w, data_symbols, pilot, FFT_length, Channel_length, fd, sampling_freq)
    h_n_total = One_Kalman_filter(pilot_symbols, variance_w, data_symbols, pilot, FFT_length, Channel_length, fd, sampling_freq)

    # Apply the channel estimation using either the Kalman filter per sub-channel or the total Kalman filter
    retrieved_data_symbols, retrieved_pilot_symbols = ofdm_equalizer(h_n_total, pilot_symbols, data_symbols)
    retrieved_data_symbols_single, retrieved_pilot_symbols_single = ofdm_equalizer(h_n_single, pilot_symbols, data_symbols)
    # Remove the zero padding of the retrieved data symbols
    retrieved_data_symbols = remove_zero_padding(retrieved_data_symbols, [110, 110])
    retrieved_data_symbols_single = remove_zero_padding(retrieved_data_symbols_single, [110, 110])

    # Assign the bits of both the pilot symbols as the data symbols from the per-subchannel Kalman filter
    bits_pilot = assign_bits(retrieved_pilot_symbols)
    bits_pilot_single = assign_bits(retrieved_pilot_symbols_single)

    # Assign the bits of both the pilot symbols as the data symbols from the total Kalman filter
    bits_data = assign_bits(retrieved_data_symbols)
    bits_data_single = assign_bits(retrieved_data_symbols_single)

    # Reshape the extracted bits to the image
    bits_data = 255 * bits_data.T.reshape(((110, 110)), order='F')
    bits_data_single = 255 * bits_data_single.T.reshape(((110, 110)), order='F')
    # Return the extracted bits
    return bits_pilot, bits_data, bits_pilot_single, bits_data_single

def calculate_bit_error_rate(retrieved_bits, pilot):
    """This function calculates the Bit error rate from the retrieved pilot bits.
    Input:
        - retrieved_bits: The retrieved pilot bits
        - pilot: The pilot symbol
    Output:
        - BER: The bit error rate"""
    # Assign bits from the true pilot symbols
    pilot_bit = assign_bits(np.array([pilot]))
    # Create the bit array from the pilot symbols
    pilot_bit = np.repeat(pilot_bit, len(retrieved_bits), axis=0)
    # Calculate the Bit Error Rate
    BER = np.sum(retrieved_bits != pilot_bit) / len(pilot_bit.flatten())
    # Return the Bit Error Rate
    return BER
