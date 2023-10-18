import numpy as np
import matplotlib.pyplot as plt
import cmath
import  math

def ofdm_equalizer(hn, pilot_symbols, data_symbols, data_index, pilot_index, plot=True):
    """Zero forcing equalization: Z =Y/H: element wise division"""
    zn_pilot = np.divide(pilot_symbols, hn)
    retrieved_data = np.zeros((7, 1000), dtype=np.complex128)
    for i in range(5):
        if i >= 3:
            pilot_assignment = data_symbols[:,i-3::5]
            zn_data = np.divide(pilot_assignment, hn)
            index = np.arange(i - 3,retrieved_data.shape[1], 5)
            retrieved_data[:,index] = zn_data

        else:
            pilot_assignment = data_symbols[:,i+2::5]
            zn_data = np.divide(pilot_assignment, hn)
            index = np.arange(i + 2,retrieved_data.shape[1], 5)
            retrieved_data[:,index] = zn_data
    if plot:
        x = [ele.real for ele in zn_data.flatten()]
        # extract imaginary part
        y = [ele.imag for ele in zn_data.flatten()]
        # plot the complex numbers
        plt.scatter(x, y,alpha=0.5)
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.show()
    return retrieved_data, zn_pilot


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
        # degrees[t] = math.degrees(cmath.phase(symbol))
        degrees[t] = np.angle(symbol)
        # for k, symbol in enumerate(sequence):
        #     degrees[t, ] = math.degrees(cmath.phase(symbol))
    return degrees

def assign_bits(retrieved_sequence):
    degrees = extract_phase(retrieved_sequence, plot=False)

    # plt.hist(degrees)
    # plt.show()
    bit_seq = []
    for t, k in enumerate(degrees):
        # bit_sub_seq = []
        # for k, symbol in enumerate(sequence):
        if 0 <= k < np.pi / 2:
            bits = [0,0]
        elif np.pi / 2 <= k < np.pi:
            bits = [0, 1]
        elif -(np.pi/2) < k <= np.pi:
            bits = [1, 1]
        else:
            bits = [1, 0]
        bit_seq.append(bits)
    return np.array(bit_seq)
