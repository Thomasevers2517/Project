"""This script contains the functions for multiple error metrics aplied to evaluete
the performance of the Kalman filters."""

# Imports
import numpy as np
from Decode import assign_bits, calculate_bit_error_rate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from Decode import extract_phase
from matplotlib.ticker import FormatStrFormatter

def iterate_over_dictionary_of_received_symbols(dict_retrieved_pilots, pilot_symbols, function_name, type='Symbol'):
    """This function iterates over the defined dictionary containing the reconstructed
    information of the pilot symbols from all the given datasets.
    Input:
        - dict_retrieved_pilots: dictionary with the retrieved information of the pilot symbols
            of all the given datasets.
        - pilot_symbols: The true pilot symbol
        - function_name: The error metric function
        - type: Whether to take the extracted symbol of bit into account. String format, either Symbol or Bits
    Output:
        - list_of_interest_total: The error of the total Kalman filter
        - list_of_interest_single: The error of the per sub-channel Kalman filter"""
    # Initiate empty lists
    list_of_interest_total = []
    list_of_interest_single = []
    # Iterate over the assigned noise levels and the retrieved pilot symbols
    for sub_dict in dict_retrieved_pilots.values():
        # Initiate empty lists
        sub_total = []
        sub_single = []
        # Iterate over 3
        for dataset_number in range(3):
            # Append the error metric of the pilot symbols retrieved by either the total Kalman filter of the
            # per sub-channel Kalman filter
            sub_total.append(function_name(sub_dict['Total'][dataset_number][type], pilot_symbols))
            sub_single.append(function_name(sub_dict['Single'][dataset_number][type], pilot_symbols))

        # Append the mean error metric
        list_of_interest_total.append(np.array(sub_total).mean())
        list_of_interest_single.append(np.array(sub_single).mean())
    # Return the lists
    return list_of_interest_total, list_of_interest_single

def plot_BER_over_Noise(dict_retrieved_pilots, pilot):
    """This function calculates the Bit Error Rate over noise.
    Input:
        - dict_retrieved_pilots: dictionary containing the retrieved pilot symbols
            under multiple noise levels.
        - pilot: The pilot symbol
    Output:
        - Total_over_Noise: The mean BER of the total Kalman filter channel estimation
        - Single_over_Noise: The mean BER of the per sub-channel Kalman filter channel estimation"""
    # Calculate the Bit error rate
    Total_over_Noise, Single_over_Noise = iterate_over_dictionary_of_received_symbols(dict_retrieved_pilots, pilot, calculate_bit_error_rate, type='Bits')
    # Print the BER values
    print('BER Total Kalman filter: ', Total_over_Noise,
    ' BER per subchannel Kalman filter: ', Single_over_Noise,
    ' Corresponding noise levels: ', list(dict_retrieved_pilots.keys()))
    # Return the mean BER erros
    return Total_over_Noise, Single_over_Noise

def root_mean_squared_error_complex_values(estimate, true_values):
    """This function calculate the Root Mean Squared error of complex values.
    Input:
        - estimate: The estimated value
        - true_values: The corresponding true value
    Output:
        The root mean squared error """
    return sqrt(np.mean(abs(estimate-true_values)**2))

def get_RMSE_over_Noise(dict_retrieved_pilots, pilot):
    """This function
    pilot_symbols = np.repeat(pilot, len(dict_retrieved_pilots[list(dict_retrieved_pilots.keys())[0]]['Total'][0]['Symbol']))
    Total_over_Noise, Single_over_Noise = iterate_over_dictionary_of_received_symbols(dict_retrieved_pilots,
    pilot_symbols, root_mean_squared_error_complex_values)
    print('RMSE Total Kalman filter: ', Total_over_Noise,
    ' RMSE per subchannel Kalman filter: ', Single_over_Noise,
    ' Corresponding noise levels: ', list(dict_retrieved_pilots.keys()))
    return Total_over_Noise, Single_over_Noise

def extract_mean(seq, pilots):
    return np.mean(seq)

def extract_std(seq, pilots):
    return np.std(seq)

def get_mean_and_std_pilot_symbols(dict_retrieved_pilots, pilot):
    mean_total, mean_single = iterate_over_dictionary_of_received_symbols(dict_retrieved_pilots,
    pilot, extract_mean)

    std_total, std_single = iterate_over_dictionary_of_received_symbols(dict_retrieved_pilots,
    pilot, extract_std)

    print('Noise level: ', list(dict_retrieved_pilots.keys()))
    print('Total Kalman Filter: ', mean_total, std_total)
    print('Per subchannel Kalman Filter: ', mean_single, std_single)

# def plot_phase_over_amount(dict_retrieved_pilots):
#
#     for keys in list(dict_retrieved_pilots.keys())[::-1]:
#         no_noise_seq_total = dict_retrieved_pilots[keys]['Total']
#         no_noise_seq_single = dict_retrieved_pilots[keys]['Single']
#         phase_radians = []
#         phase_radians_single = []
#         for dataset_number in range(3):
#             phase_radians.append(extract_phase(no_noise_seq_total[dataset_number]['Symbol']))
#             phase_radians_single.append(extract_phase(no_noise_seq_single[dataset_number]['Symbol']))
#         fig, (ax1, ax2) = plt.subplots(1,2)
#         ax1.hist(np.array(phase_radians).flatten(), label=keys, edgecolor='black', bins=30)
#         ax1.ticklabel_format(useOffset=False)
#         ax1.set_title('Total Kalman filter')
#         ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         # ax1.set_xlim(0, 1.5)
#
#         ax2.hist(np.array(phase_radians_single).flatten(), label=keys, edgecolor='black', bins=30)
#         ax2.ticklabel_format(useOffset=False)
#         ax2.set_title('Kalman filter per sub-channel')
#         ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         # ax2.set_xlim(0, 1.5)
#
#         fig.suptitle('Histogram of extracted symbols from {}'.format(keys))
#         fig.tight_layout()
#         plt.show()
