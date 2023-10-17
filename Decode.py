import numpy as np
import matplotlib.pyplot as plt
import cmath
import  math

def ofdm_equalizer(hn, pilot_symbols, data_symbols, data_index, pilot_index, plot=True):
    """Zero forcing equalization: Z =Y/H: element wise division"""
    zn_pilot = np.divide(pilot_symbols, hn)
    pilot_assignment = np.zeros((5,7,200), dtype=np.complex128).T
    zn_data = np.zeros((5, 7, 200), dtype=np.complex128)
    retrieved_data_symbols = np.zeros((7, 1000), dtype=np.complex128).T
    corresponding_data_symbols = []
    counter = 0
    for subcarrier in pilot_index[0][0][0]:

        # calculate the difference array
        difference_array = np.absolute(data_index[0][0][0]-subcarrier)

        # find the index of minimum element from the array
        index = difference_array.argmin()
        if index != 0:
            indexed_subchannels = data_index[0][0][0][index-3:index+2]

            pilot_assignment[counter] = data_symbols[:, index-3:index+2]
        else:
            indexed_subchannels = data_index[0][0][0][index:index+2]
            pilot_assignment[counter] = np.concatenate([np.zeros((7,3), dtype=np.complex128),data_symbols[:, index:index+2]], axis=1)

        counter += 1
    i_total = 0
    for i in pilot_assignment.T:
        zn_data[i_total] = np.divide(i, hn)
        i_total += 1

    initial_index = 0
    for j in zn_data.T:
        counter = initial_index
        for k in j.T:
            retrieved_data_symbols[counter] = k
            counter += 5

        initial_index += 1
    retrieved_data_symbols = retrieved_data_symbols.T

    if plot:
        # extract real part
        x = [ele.real for ele in retrieved_data_symbols]
        # extract imaginary part
        y = [ele.imag for ele in retrieved_data_symbols]

        # plot the complex numbers
        plt.scatter(x, y)
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.show()
    return retrieved_data_symbols, zn_pilot


def remove_zero_padding(retrieved_data_symbols, data_indices, image_shape):
    """Start index zero padding later than final symbol (ofcourse), no removing needed
    Use chanel to predict whole 2048 sequence?"""
    # print(retrieved_data_symbols.shape)
    # data_indices = data_indices[0][0][0]
    # start_index_final_symbol = 160+(7*160)+(6*2048)
    # start_index_zero_padding = 15570
    # print(data_indices+start_index_final_symbol)
    seq_length = image_shape[0] * image_shape[1]
    total_seq = []
    for sub_sequence in retrieved_data_symbols:
        total_seq.extend(sub_sequence)
    total_length = 2* len(total_seq)
    zero_padding_len = int((total_length - seq_length)/2)
    no_padding_seq = np.array(total_seq)[:-zero_padding_len]
    return no_padding_seq

def flatten_pilot_symbols(retrieved_pilot_sequence):
    total_seq = []
    for sub_sequence in retrieved_pilot_sequence:
        total_seq.extend(sub_sequence)
    return np.array(total_seq)

def extract_phase(retrieved_sequence, plot=False):
    degrees=np.zeros((retrieved_sequence.shape))
    for t, symbol in enumerate(retrieved_sequence):
        degrees[t] = math.degrees(cmath.phase(symbol))
        # for k, symbol in enumerate(sequence):
        #     degrees[t, ] = math.degrees(cmath.phase(symbol))
    return degrees

def assign_bits(retrieved_sequence):
    degrees = extract_phase(retrieved_sequence, plot=False)
    bit_seq = []
    for t, k in enumerate(degrees):
        # bit_sub_seq = []
        # for k, symbol in enumerate(sequence):
        if 0 <= k < 90:
            bits = [0,0]
        elif 90 <= k < 180:
            bits = [0, 1]
        elif 180 <= k < 270:
            bits = [1, 1]
        elif 270 <= k:
            bits = [1, 0]
        bit_seq.append(bits)
    return np.array(bit_seq)
