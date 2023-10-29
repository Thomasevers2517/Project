"""This is the main script containing the combination of all the necessary functions"""

# Imports
import scipy
from Decode import extract_received_signal
from Evaluation import calculate_bit_error_rate, plot_BER_over_Noise, get_RMSE_over_Noise, plot_phase_over_amount,get_mean_and_std_pilot_symbols
import matplotlib.pyplot as plt

"""Reconstruct single image"""
matfile = scipy.io.loadmat("../DataSet_OFDM/New_DataSet/DataSet1.mat")
signal = matfile['HighNoise_RxSignal']
variance_w = 0.1 # 0 for no noise, -25 dB for (0.005) low noise and -10 dB for high noise (0.1))
pilot = matfile['OFDM']['PilotSymbol'][0][0][0]
Channel_length = matfile['Channel']['Length'][0][0][0][0]
FFT_length = matfile['OFDM']['FFT_Length'][0][0][0][0]

"""Reconstruct all images and save bits as a dictionary"""
dict_retrieved_pilots = {'NoNoise_RxSignal': {'Total': [], 'Single': []}, 'LowNoise_RxSignal': {'Total': [], 'Single': []}, 'HighNoise_RxSignal': {'Total': [], 'Single': []}}
for i in range(1,4):
    mat = scipy.io.loadmat("../DataSet_OFDM/New_DataSet/DataSet{}.mat".format(str(i)))
    for signal_type, variance_w in zip(dict_retrieved_pilots.keys(), [0, 0.005, 0.1]):
        signal = matfile[signal_type]
        bits_pilot, retrieved_pilot_symbols, bits_data, bits_pilot_single, retrieved_pilot_symbols_single, bits_data_single = extract_received_signal(signal, variance_w, pilot, Channel_length, FFT_length, mat, combiner=True)
        dict_retrieved_pilots[signal_type]['Total'].append({'Bits': bits_pilot, 'Symbol': retrieved_pilot_symbols})
        dict_retrieved_pilots[signal_type]['Single'].append({'Bits':bits_pilot_single, 'Symbol':retrieved_pilot_symbols_single})

"""Reconstruct single image using the definitions above"""
bits_pilot, retrieved_pilot_symbols, bits_data, bits_pilot_single, retrieved_pilot_symbols, bits_data_single = extract_received_signal(signal, variance_w, pilot, Channel_length, FFT_length, matfile, combiner=True)
BER = calculate_bit_error_rate(bits_pilot, pilot)

"""Show the single selected image"""
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('retrieved images')
# ax1.imshow(bits_data)
# ax1.set_title('Total Kalman Filter')
# ax2.imshow(bits_data_single)
# ax2.set_title('Kalman filter per subchannel')
# plt.show()

"""Plot Bit Error Rate over Noise of all the generated images"""
# BER_Total, BER_Single = plot_BER_over_Noise(dict_retrieved_pilots, pilot)
# RMSE_Total, RMSE_Single = get_RMSE_over_Noise(dict_retrieved_pilots, pilot)
get_mean_and_std_pilot_symbols(dict_retrieved_pilots, pilot)
# plot_phase_over_amount(dict_retrieved_pilots)
