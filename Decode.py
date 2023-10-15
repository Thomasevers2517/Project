import numpy as np
import matplotlib.pyplot as plt

def ofdm_equalizer(hn, pilot_symbols, data_symbols, data_index, pilot_index, plot=True):
    """Zero forcing equalization: Z =Y/H: element wise division"""
    zn_pilot = np.divide(pilot_symbols, hn)


    # for subcarrier in pilot_index[0][0][0]:
    #     print("Value to which nearest element is to be found: ", subcarrier)
    #
    #     # calculate the difference array
    #     difference_array = np.absolute(data_index[0][0][0]-subcarrier)
    #
    #     # find the index of minimum element from the array
    #     index = difference_array.argmin()
    #     if index != 0:
    #         indexed_subchannels = data_index[0][0][0][index-3:index+2]
    #     else:
    #         indexed_subchannels = data_index[0][0][0][index:index+2]
    #     # for indexed_data_subchannel in indexed_subchannels:
    #
    #     print("Index of nearest value is : ", index)

    if plot:
        # extract real part
        x = [ele.real for ele in zn_pilot]
        # extract imaginary part
        y = [ele.imag for ele in zn_pilot]

        # plot the complex numbers
        plt.scatter(x, y)
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.show()
