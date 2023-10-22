"""This script contains the evaluation function to create the BER plot over noise
of the pilot symbols"""

# Imports
import numpy as np
from Decode import assign_bits, calculate_bit_error_rate
import matplotlib.pyplot as plt

def plot_BER_over_Noise(dict_retrieved_pilots, pilot):
    """This function plots the Bit Error Rate over noise.
    Input:
        - dict_retrieved_pilots: dictionary containing the retrieved pilot symbols
            under multiple noise levels.
        - pilot: The pilot symbol
    Output:
        - Total_over_Noise: The mean BER of the total Kalman filter channel estimation
        - Single_over_Noise: The mean BER of the per sub-channel Kalman filter channel estimation"""
    # Initiate empty lists
    Total_over_Noise = []
    Single_over_Noise = []
    NoiseLevels = []
    # Iterate over the assigned noise levels and the retrieved pilot symbols
    for noise_level, sub_dict in zip(dict_retrieved_pilots.keys(), dict_retrieved_pilots.values()):
        # Append the noise level
        NoiseLevels.append(noise_level)
        # Initiate empty lists
        sub_total_BER = []
        sub_single_BER = []
        # Iterate over 3
        for i in range(3):
            # Append the bit error of the pilot symbols retrieved by either the total Kalman filter of the
            # per sub-channel Kalman filter
            sub_total_BER.append(calculate_bit_error_rate(sub_dict['Total'][i], pilot))
            sub_single_BER.append(calculate_bit_error_rate(sub_dict['Total'][i], pilot))
        # Append the mean BER
        Total_over_Noise.append(np.array(sub_total_BER).mean())
        Single_over_Noise.append(np.array(sub_single_BER).mean())
    # Plot the BER over noise
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Mean Bit Error Rate Pilot Samples')
    ax1.plot(NoiseLevels, Total_over_Noise)
    ax1.set_title('BER Total Kalman Filter')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 90)
    ax2.plot(NoiseLevels, Single_over_Noise)
    ax2.set_title('BER Kalman filter per subchannel')
    ax2.set_xticklabels(ax1.get_xticklabels(), rotation = 90)
    fig.tight_layout()
    plt.show()
    # Return the mean BER erros
    return Total_over_Noise, Single_over_Noise
