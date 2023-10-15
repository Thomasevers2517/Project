import cmath
import scipy.io
import numpy as np
from Kalman import One_Kalman_filter, Kalman_filter_per_channel
from Decode import ofdm_equalizer

def find_prefix(sequence, subseq_length):
    cyclic_prefix = dict(prefix_start = [], reoccurence = [])
    print(sequence.shape)
    for prefix_start_point in range(len(sequence) - subseq_length):
        subsequence = sequence[prefix_start_point:prefix_start_point + subseq_length]
        start_point = prefix_start_point
        while start_point < len(sequence) - subseq_length:
            selected_seq = sequence[start_point:start_point+subseq_length]
            if np.array_equal(selected_seq, subsequence) and prefix_start_point != start_point:
                if prefix_start_point != start_point:
                    cyclic_prefix['prefix_start'].append(prefix_start_point)
                    cyclic_prefix['reoccurence'].append(start_point)
                    # print(prefix_start_point, start_point, subsequence.shape)
            start_point += 1
    return cyclic_prefix

def remove_cyclic_prefix(signal, length):
    sub_signal = []
    for prefix in range(0, len(signal), 160+2048):
        start_cyclic_prefix = prefix
        end_cyclic_prefix = prefix + 160
        sub = signal[end_cyclic_prefix:end_cyclic_prefix+2048]
        if len(sub) == 2048:
            sub_signal.append(sub)
    return np.array(sub_signal)

def to_frequency_domain_subsequences(sequence):
    sub_sequences = remove_cyclic_prefix(sequence, 46)
    freq_list = []
    for sub in sub_sequences:
        freq_sub = np.fft.fft(sub.T)
        freq_list.append(freq_sub)
    return np.array(freq_list)

def extract_pilot_symbols(freq_domain_signals, matfile):
    pilot_indices = matfile['OFDM']['PilotIndices']
    pilots = []
    for sample in frequency_domain_signals:
        pilot_symbols = sample[0][pilot_indices[0][0][0]]
        pilots.append(pilot_symbols)
    return np.array(pilots), pilot_indices

def extract_data_symbols(frequency_domain_signals, matfile):
    data_indices = matfile['OFDM']['DataIndices']
    data = []
    for sample in frequency_domain_signals:
        data_symbols = sample[0][data_indices[0][0][0]]
        data.append(data_symbols)
    return np.array(data), data_indices

def extract_all_symbols(freq_domain_signals, matfile):
    pilot_indices = matfile['OFDM']['PilotIndices'][0][0][0]
    data_indices = matfile['OFDM']['DataIndices'][0][0][0]
    symbols = []
    c = np.concatenate((pilot_indices,data_indices))
    c = np.sort(c)
    for sample in frequency_domain_signals:
        symbol = sample[0][c]
        symbols.append(symbol)
    return np.array(symbols), c

matfile = scipy.io.loadmat('../DataSet_OFDM/New_DataSet/DataSet2.mat')

No_Noise_signal = matfile['NoNoise_RxSignal']
variance_w = 0 # 0 for no noise, -25 dB for low noise and -10 dB for high noise)


frequency_domain_signals = to_frequency_domain_subsequences(No_Noise_signal)
pilot_symbols, pilot_indices = extract_pilot_symbols(frequency_domain_signals, matfile)
data_symbols, data_indices = extract_data_symbols(frequency_domain_signals, matfile)
all_symbols, all_indices = extract_all_symbols(frequency_domain_signals, matfile)

h_n_single = Kalman_filter_per_channel(pilot_symbols, variance_w, data_symbols, plot=False)
h_n_total = One_Kalman_filter(pilot_symbols, variance_w, data_symbols, plot=False)

ofdm_equalizer(h_n_single, pilot_symbols, data_symbols, data_indices, pilot_indices, plot=True)
