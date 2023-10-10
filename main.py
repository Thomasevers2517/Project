import scipy.io
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#from Kalman import filter_sequence, retrieve_data_symbols

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

        if np.array_equal(signal[start_cyclic_prefix:end_cyclic_prefix] , sub[-160:]):
            raise Exception("Not identical cyclic prefix")
        
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




matfile = scipy.io.loadmat('./DataSet_OFDM/New_DataSet/DataSet1.mat')

No_Noise_signal = matfile['HighNoise_RxSignal']

frequency_domain_signals = to_frequency_domain_subsequences(No_Noise_signal)
pilot_symbols, pilot_indices = extract_pilot_symbols(frequency_domain_signals, matfile)
data_symbols, data_indices = extract_data_symbols(frequency_domain_signals, matfile)

np.set_printoptions(threshold=100)
# symbols =  7 symbos for each carrier
print("Pilot symbols, data symbols, data indices, pilot indices ", pilot_symbols.shape, data_symbols.shape, np.shape(data_indices), np.shape(pilot_indices))


test = 1/1j
if test != -1j:
    raise Exception("Not equal to -1j")

transmited_pilot = 0.707 + 0.707j
Sigma = np.full(200, None, dtype=np.complex128)
P = np.zeros((7,200), dtype=np.complex128)
C = np.zeros((7,200), dtype=np.complex128)
M_n = np.zeros((7,200), dtype=np.complex128)
lambda_ = np.zeros((7,200), dtype=np.complex128)
K = np.zeros((7,200), dtype=np.complex128)
variance_Z = np.zeros((7,200), dtype=np.complex128)

print("transmited pilot", transmited_pilot)
h = np.zeros((7,200), dtype=np.complex128)
auto_correlation = np.zeros((7-1,200), dtype=np.complex128)
for t, pilot_symbols_at_T in enumerate(pilot_symbols):
    if t < 2:
        h[t] = pilot_symbols_at_T / transmited_pilot
    else:
        for j in range(h.shape[1]):
            subchannel_h = h[:, j]
            if j ==123:
                print(f"subchannel_h 123: \n, {subchannel_h[0:t]}")
                print(F"Type subchannel 123: {type(subchannel_h[0])}")
            if subchannel_h.shape[0] > 7:
                raise Exception("Subchannel length is more than 7")
            auto_correlation[:t, j] = np.correlate(subchannel_h[0:t], subchannel_h[0:t], mode='full')[:len(subchannel_h[0:t])]
        if t==6:
            print(f"auto correlation 123: \n {auto_correlation[:,123]} \n auto corr 195:  \n {auto_correlation[:,195]}") # for 3 subchannels, print the autocorrelation for K>7
        P[t] = auto_correlation[1]/auto_correlation[0]
        variance_Z[t] = (auto_correlation[0]**2 - abs(auto_correlation[1])**2) / auto_correlation[0]
        if Sigma[0] == None:
            Sigma = auto_correlation[0]
        C[t] = -P[t]
        M_n[t] = C[t] * Sigma * np.conj(C[t]) + 1
        variance_omega = 0 # noise variance
        lambda_[t] =  transmited_pilot * M_n[t] * np.conj(transmited_pilot) + variance_omega
        K[t] = M_n[t] * transmited_pilot / lambda_[t]
        h[t] = C[t] * h[t-1] + K[t] *(pilot_symbols_at_T - transmited_pilot * h[t-1])

        h[t] = 0.5*h[t-1] + 0.5 * pilot_symbols_at_T / transmited_pilot
        Sigma = (1-K[t]*transmited_pilot) * M_n[t]

plt.imshow(np.abs(h))
plt.colorbar()
plt.show()
        # h_minus = P * h[-1] + Z




# measured_pilot = np.full((7, 200), abs(0.707106781186548 + 0.707106781186548j))
# Kalman, smoothed_kf = filter_sequence(pilot_symbols, measured_pilot)
# retrieved_data = retrieve_data_symbols(Kalman, frequency_domain_signals, data_indices[0][0][0])
# print(retrieved_data)
