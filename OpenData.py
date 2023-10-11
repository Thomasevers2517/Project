import cmath 
import scipy.io
import numpy as np
# from Kalman import KalmanFilter_all_pilots
import matplotlib.pyplot as plt
from scipy.special import j0

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

matfile = scipy.io.loadmat('./DataSet_OFDM/New_DataSet/DataSet1.mat')

No_Noise_signal = matfile['HighNoise_RxSignal']

# matfile = scipy.io.loadmat('./DataSet_OFDM/New_DataSet/DataSet1.mat')
#
# No_Noise_signal = matfile['HighNoise_RxSignal']

frequency_domain_signals = to_frequency_domain_subsequences(No_Noise_signal)
pilot_symbols, pilot_indices = extract_pilot_symbols(frequency_domain_signals, matfile)
data_symbols, data_indices = extract_data_symbols(frequency_domain_signals, matfile)
all_symbols, all_indices = extract_all_symbols(frequency_domain_signals, matfile)
# KalmanFilter_all_pilots(pilot_symbols)

np.set_printoptions(threshold=100)
# symbols =  7 symbos for each carrier
print("Pilot symbols, data symbols, data indices, pilot indices ", pilot_symbols.shape, data_symbols.shape, np.shape(data_indices), np.shape(pilot_indices))


test = 1/1j
if test != -1j:
    raise Exception("Not equal to -1j")

transmited_pilot = 0.707 + 0.707j
Sigma_n = np.zeros((7,200), dtype=np.complex128)
Sigma_prev = np.zeros(200, dtype=np.complex128) # We kunnen dit denk ik beter initialiseren
A = np.zeros((7,200), dtype=np.complex128) # Wat is P ten opzichte van Sigma -> A?
C = np.zeros((7,200), dtype=np.complex128)
M_n = np.zeros((7,200), dtype=np.complex128)
lambda_ = np.zeros((7,200), dtype=np.complex128)
K = np.zeros((7,200), dtype=np.complex128)
xn = np.zeros((7,200), dtype=np.complex128)
# variance_Z = np.zeros((7,200), dtype=np.complex128)
a = 0.5

print("transmited pilot", transmited_pilot)
h = np.zeros((7,200), dtype=np.complex128)
####
pilot = 0.707 + 0.707j
a = -a
C = a
variance_w = 0 # 0 for no noise, -25 dB for low noise and -10 dB for high noise)
T, fd = 0.01, 100
auto_correlation = [j0(2*np.pi*fd * 0 * T), j0(2*np.pi*fd * 1 *T)] # By equation from paper
Sigma_prev = (auto_correlation[0]**2 - abs(auto_correlation[1])**2) / auto_correlation[0]
xn_prev = 1
g = 1 # Not correct yet?

for t, symbols_at_T in enumerate(pilot_symbols.T):
    
    M_n[:,t] = C * Sigma_prev * np.conjugate(C) + 1
    lambda_[:,t] = pilot * M_n[:,t] * np.conjugate(pilot) + variance_w
    K[:,t] = M_n[:,t] * np.conjugate(pilot) / lambda_[:,t]
    xn[:,t] = C*xn_prev + K[:,t] * (symbols_at_T - pilot*C* xn_prev)
    Sigma_n[:,t] = (1- K[:,t] * pilot) * M_n[:,t]
    xn_prev = xn[:,t]
    Sigma_prev = Sigma_n[:,t]


fig, axs = plt.subplots(6)
im0 = axs[0].imshow(abs(K))
im1 = axs[1].imshow(abs(M_n))
im2 = axs[2].imshow(abs(lambda_))
im3 = axs[3].imshow(abs(Sigma_n))
im4 = axs[4].imshow(abs(pilot_symbols/xn))
im5 = axs[5].imshow(np.angle(xn))
fig.colorbar(im0, ax=axs[0])
fig.colorbar(im1, ax=axs[1])
fig.colorbar(im2, ax=axs[2])
fig.colorbar(im3, ax=axs[3])
fig.colorbar(im4, ax=axs[4])
plt.show()

# plt.imshow(np.abs(h))
# plt.colorbar()
# plt.show()
