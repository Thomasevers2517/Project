import cmath
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from Kalman import One_Kalman_filter, Kalman_filter_per_channel
from Decode import ofdm_equalizer, remove_zero_padding, assign_bits, flatten_pilot_symbols, calculate_bit_error_rate

def remove_cyclic_prefix(signal, length, FFT_length):
    sub_signal = []
    for prefix in range(0, len(signal), length+FFT_length):
        start_cyclic_prefix = prefix
        end_cyclic_prefix = prefix + length
        sub = signal[end_cyclic_prefix:end_cyclic_prefix+FFT_length]
        if len(sub) == FFT_length:
            sub_signal.append(sub)
    return np.array(sub_signal)

def to_frequency_domain_subsequences(sequence, Channel_length, FFT_Length):
    sub_sequences = remove_cyclic_prefix(sequence, Channel_length - 2, FFT_Length)
    freq_list = []
    for sub in sub_sequences:
        sub = np.reshape(sub, (FFT_length,))
        freq_sub = np.fft.fft(sub)
        freq_list.append(freq_sub)
    return np.array(freq_list)

def extract_pilot_symbols(frequency_domain_signals, matfile):
    pilot_indices = matfile['OFDM']['PilotIndices']-1
    pilots = []
    for sample in frequency_domain_signals:
        pilot_symbols = sample[pilot_indices[0][0][0]]
        pilots.append(np.reshape(np.array(pilot_symbols), (200,)))
    return np.array(pilots), pilot_indices

def extract_data_symbols(frequency_domain_signals, matfile):
    data_indices = matfile['OFDM']['DataIndices']-1
    data = []
    for sample in frequency_domain_signals:
        data_symbols = sample[data_indices[0][0][0]]
        data.append(np.reshape(np.array(data_symbols), (1000,)))
    return np.array(data), data_indices

def extract_received_signal(signal, variance_w, pilot, Channel_length, FFT_length, matfile):
    frequency_domain_signals = to_frequency_domain_subsequences(signal, Channel_length, FFT_length)
    pilot_symbols, pilot_indices = extract_pilot_symbols(frequency_domain_signals, matfile)
    data_symbols, data_indices = extract_data_symbols(frequency_domain_signals, matfile)

    h_n_single = Kalman_filter_per_channel(pilot_symbols, variance_w, data_symbols, plot=False)
    h_n_total = One_Kalman_filter(pilot_symbols, variance_w, data_symbols, plot=False)

    retrieved_data_symbols, retrieved_pilot_symbols = ofdm_equalizer(h_n_total, pilot_symbols, data_symbols, data_indices, pilot_indices, plot=False)
    retrieved_data_symbols_single, retrieved_pilot_symbols_single = ofdm_equalizer(h_n_single, pilot_symbols, data_symbols, data_indices, pilot_indices, plot=False)
    retrieved_data_symbols = remove_zero_padding(retrieved_data_symbols, data_indices, [110, 110])
    retrieved_data_symbols_single = remove_zero_padding(retrieved_data_symbols_single, data_indices, [110, 110])

    # retrieved_pilot_symbols = flatten_pilot_symbols(retrieved_pilot_symbols)
    # retrieved_pilot_symbols_single = flatten_pilot_symbols(retrieved_pilot_symbols_single)

    bits_pilot = assign_bits(retrieved_pilot_symbols)
    bits_pilot_single = assign_bits(retrieved_pilot_symbols_single)

    bits_data = assign_bits(retrieved_data_symbols)
    bits_data_single = assign_bits(retrieved_data_symbols_single)

    bits_data = 255*bits_data.T.reshape(((110, 110)), order='F')
    bits_data_single = 255 * bits_data_single.T.reshape(((110, 110)), order='F')
    return bits_pilot, bits_data, bits_pilot_single, bits_data_single

matfile = scipy.io.loadmat("../DataSet_OFDM/New_DataSet/DataSet1.mat")
signal = matfile['HighNoise_RxSignal']
variance_w = 0.1 # 0 for no noise, -25 dB for (0.005) low noise and -10 dB for high noise (0.1))
pilot = matfile['OFDM']['PilotSymbol']
Channel_length = matfile['Channel']['Length'][0][0][0][0]
FFT_length = matfile['OFDM']['FFT_Length'][0][0][0][0]

bits_pilot, bits_data, bits_pilot_single, bits_data_single = extract_received_signal(signal, variance_w, pilot, Channel_length, FFT_length, matfile)
# calculate_bit_error_rate(bits_pilot, pilot)



fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('retrieved images')
ax1.imshow(bits_data)
ax1.set_title('Total Kalman Filter')
ax2.imshow(bits_data_single)
ax2.set_title('Kalman filter per subchannel')
plt.show()
