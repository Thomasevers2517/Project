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

matfile = scipy.io.loadmat('./DataSet_OFDM/New_DataSet/DataSet2.mat')

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


print("transmited pilot", transmited_pilot)
####
pilot = 0.707 + 0.707j

variance_w = 0.1 # 0 for no noise, -25 dB for low noise and -10 dB for high noise)
T, fd = (2048+160)/30720000, 50
auto_correlation = [j0(2*np.pi*fd * 0 * T), j0(2*np.pi*fd * 1 *T)] # By equation from paper
Sigma_prev = np.full((200),(auto_correlation[0]**2 - abs(auto_correlation[1])**2) / auto_correlation[0])

a = auto_correlation[1] / auto_correlation[0]
a = -a
C = a

print("Sigma_prev", Sigma_prev)

xn_prev = np.full((200) ,0)
g = 1 # Not correct yet?

for t, symbols_at_T in enumerate(pilot_symbols):
    
    M_n[t,:] = C * Sigma_prev * np.conjugate(C) + 1
    if M_n[t,:].shape != (200,):
        raise Exception(f"M_N at T: {M_n[t,:].shape}")
    
    lambda_[t,:] = pilot * M_n[t,:] * np.conjugate(pilot) + variance_w

    K[t,:] = M_n[t,:] * np.conjugate(pilot) / lambda_[t,:]

    if symbols_at_T.shape != (200,):
        raise Exception(f"Symbols at T: {symbols_at_T.shape}")
    
    xn[t,:] = C*xn_prev + K[t,:] * (symbols_at_T - pilot*C* xn_prev)
    Sigma_n[t,:] = (1- K[t,:] * pilot) * M_n[t,:]
    xn_prev = xn[t,:]
    Sigma_prev = Sigma_n[t,:]


print("M_n", M_n[0,:])
print("lambda_", lambda_[0,:])
print("K", K[0,:])
print("xn", xn[0,:])
print("Sigma_n", Sigma_n[0,:])
fig_1, axs_1 = plt.subplots(4, figsize=(10, 8))
im0 = axs_1[0].imshow(abs(K), aspect='auto')
im1 = axs_1[1].imshow(abs(M_n), aspect='auto')
im2 = axs_1[2].imshow(abs(lambda_), aspect='auto')
im3 = axs_1[3].imshow(abs(Sigma_n), aspect='auto')

axs_1[0].set_title('Kalman filter coefficients')
axs_1[1].set_title('M_n')
axs_1[2].set_title('lambda')
axs_1[3].set_title('Sigma_n')
fig_1.colorbar(im0, ax=axs_1[0], shrink=0.8)
fig_1.colorbar(im1, ax=axs_1[1], shrink=0.8)
fig_1.colorbar(im2, ax=axs_1[2], shrink=0.8)
fig_1.colorbar(im3, ax=axs_1[3], shrink=0.8)

fig_2, axs_2 = plt.subplots(3,3, figsize=(10, 8))

im4 = axs_2[0, 0].imshow(abs(data_symbols[:,::5]/xn[:]), aspect='auto')
im5 = axs_2[1, 0].imshow(np.angle(data_symbols[:,::5]/xn[ :]))
im6 = axs_2[2, 0].imshow(abs(xn), aspect='auto')
im7 = axs_2[0, 1].imshow(np.angle(xn), aspect='auto')
im8 = axs_2[1, 1].imshow(abs(data_symbols), aspect='auto' )
im9 = axs_2[2, 1].imshow(np.angle(data_symbols), aspect='auto')
im10 = axs_2[0, 2].imshow(abs(pilot_symbols), aspect='auto')
im11 = axs_2[1, 2].imshow(np.angle(pilot_symbols), aspect='auto')

axs_2[0, 0].set_title('|data_symbols[:,::5]/xn[:]|')
axs_2[1, 0].set_title('arg(data_symbols[:,::5]/xn[:])')
axs_2[2, 0].set_title('|xn|')
axs_2[0, 1].set_title('arg(xn)')
axs_2[1, 1].set_title('|data_symbols|')
axs_2[2, 1].set_title('arg(data_symbols)')
axs_2[0, 2].set_title('|pilot_symbols|')
axs_2[1, 2].set_title('arg(pilot_symbols)')

fig_2.colorbar(im4, ax=axs_2[0, 0], shrink=0.8, aspect=40)
fig_2.colorbar(im5, ax=axs_2[1, 0], shrink=0.8, aspect=40)
fig_2.colorbar(im6, ax=axs_2[2, 0], shrink=0.8, aspect=40)
fig_2.colorbar(im7, ax=axs_2[0, 1], shrink=0.8, aspect=40)
fig_2.colorbar(im8, ax=axs_2[1, 1], shrink=0.8, aspect=40)
fig_2.colorbar(im9, ax=axs_2[2, 1], shrink=0.8, aspect=40)
fig_2.colorbar(im10, ax=axs_2[0, 2], shrink=0.8, aspect=40)
fig_2.colorbar(im11, ax=axs_2[1, 2], shrink=0.8, aspect=40)

for ax in axs_1.flat:

    ax.set_ylabel('Time')
    ax.set_ylim([0, data_symbols.shape[0]])

for ax in axs_2.flat:
   
    ax.set_ylabel('Time')
    ax.set_ylim([0, data_symbols.shape[0]])

plt.show()