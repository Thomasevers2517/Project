import scipy.io
import numpy as np
# from Kalman import KalmanFilter_all_pilots
import matplotlib.pyplot as plt

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

matfile = scipy.io.loadmat('../DataSet_OFDM/New_DataSet/DataSet1.mat')

No_Noise_signal = matfile['NoNoise_RxSignal']

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
Sigma_n = np.full(200, None, dtype=np.complex128)
Sigma_prev = np.full(200, None, dtype=np.complex128) # We kunnen dit denk ik beter initialiseren
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
# auto_correlation = np.zeros((7-1,200), dtype=np.complex128)
####
auto_correlation = np.cov((pilot_symbols), dtype=np.complex128)
####
y = 0.707 + 0.707j

for subchannel in pilot_symbols.T:
    a = -a
    C = a
    variance_w = 0 # 0 for no noise, -25 dB for low noise and -10 dB for high noise)
    auto_correlation = np.correlate(subchannel, subchannel, mode='full')[:7] # By equation from paper
    Sigma_prev = (auto_correlation[0]**2 - abs(auto_correlation[1])**2) / auto_correlation[0]
    Sigma_prev = 10
    xn_prev = 0
    g = 1 # Not correct yet?
    for t in range(0, len(subchannel)):
        M_n[t] = np.dot(np.dot(C, Sigma_prev), np.conjugate(C)) + 1
        lambda_[t] = np.dot(np.dot(y, M_n[t]), np.conjugate(y)) + variance_w
        K[t] = np.dot(M_n[t], np.conjugate(y)) / lambda_[t]
        xn[t] = np.dot(C, xn_prev) + np.dot(K[t], (subchannel[t] - (np.dot(np.dot(y,C), xn_prev))))
        Sigma_n[t] = np.dot((1- np.dot(K[t], y)), M_n[t])
        xn_prev = xn[t]
        Sigma_prev = Sigma_n[t]

    fig, axs = plt.subplots(6)
    axs[0].plot(K)
    axs[1].plot(M_n)
    axs[2].plot(lambda_)
    axs[3].plot(Sigma_n)
    axs[4].plot(Sigma_prev)
    axs[5].plot(xn)

    plt.show()

    break
# plt.imshow(np.abs(h))
# plt.colorbar()
# plt.show()





























#
# # for t, pilot_symbols_at_T in enumerate(pilot_symbols):
# """Ik denk als we dit omschrijven naar matrices dat het leesbaarder wordt en beter te handelen. Lekker bezig """
# for t, pilot_symbols_at_T in enumerate(all_symbols):
#     if t < 2:
#         h[t] = pilot_symbols_at_T / transmited_pilot # Deze snap ik ook niet.
#     else:
#         for j in range(h.shape[1]):
#             subchannel_h = h[:, j]
#             if j ==123:
#                 print(f"subchannel_h 123: \n, {subchannel_h[0:t]}")
#                 print(F"Type subchannel 123: {type(subchannel_h[0])}")
#             if subchannel_h.shape[0] > 7:
#                 raise Exception("Subchannel length is more than 7")
#             auto_correlation[:t, j] = np.correlate(subchannel_h[0:t], subchannel_h[0:t], mode='full')[:len(subchannel_h[0:t])] # Correlate is wat anders dan covariance toch? Misschien die formule uit de paper toepassen
#         if t==6:
#             print(f"auto correlation 123: \n {auto_correlation[:,123]} \n auto corr 195:  \n {auto_correlation[:,195]}") # for 3 subchannels, print the autocorrelation for K>7
#         P[t] = auto_correlation[1]/auto_correlation[0] # Deze snap ik ook niet helemaal. Is P niet 0.99*h[t-1]
#         variance_Z[t] = (auto_correlation[0]**2 - abs(auto_correlation[1])**2) / auto_correlation[0] # Deze wordt niet weer gebruikt, wat bedoelde je hiermee?
#         if Sigma[0] == None:
#             Sigma = auto_correlation[0]
#         C[t] = -P[t] # C[t] snap ik hier niet helemaal. Dit is toch de transition matrix
#         M_n[t] = C[t] * Sigma * np.conj(C[t]) + 1
#         variance_omega = 0 # noise variance
#         lambda_[t] =  transmited_pilot * M_n[t] * np.conj(transmited_pilot) + variance_omega
#         K[t] = M_n[t] * transmited_pilot / lambda_[t] # geen / maar inverse
#         h[t] = C[t] * h[t-1] + K[t] *(pilot_symbols_at_T - transmited_pilot  * h[t-1]) # Volgens mij moet hier niet transmited_pilot toch?
#
#         h[t] = 0.5*h[t-1] + 0.5 * pilot_symbols_at_T / transmited_pilot # Hier ben ik een beetje lost
#         Sigma = (1-K[t]*transmited_pilot) * M_n[t] # M_n[t-1] ipv M_n[t]
#
#
# print(h[0][pilot_indices[0][0][0]-425] * all_symbols[0][pilot_indices[0][0][0]-425])
# plt.imshow(np.abs(h))
# plt.colorbar()
# plt.show()
        # h_minus = P * h[-1] + Z

# frequency_domain_signals = to_frequency_domain_subsequences(No_Noise_signal)
# pilot_symbols, pilot_indices = extract_pilot_symbols(frequency_domain_signals, matfile)
# data_symbols, data_indices = extract_data_symbols(frequency_domain_signals, matfile)

#
# print(all_symbols.shape)
# Kalman, real_symbols, imaginary_symbols = filter_sequence(all_symbols)
# retrieved_data = retrieve_data_symbols(real_symbols, imaginary_symbols)
# print(retrieved_data.shape)
# print(retrieved_data[:, pilot_indices[0][0][0]-425])
