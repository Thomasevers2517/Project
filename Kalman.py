import numpy as np
from scipy.special import j0
import matplotlib.pyplot as plt
from scipy import signal

def One_Kalman_filter(pilot_symbols, variance_w, data_symbols, plot=True):
    sigma_t = 25 # Check max delay spread late
    pilot = 0.707 + 0.707j
    variance_w = 0 # 0 for no noise, -25 dB for low noise and -10 dB for high noise)
    T, fd = (2048+160)/30720000, 50

    Sigma_prev = np.zeros((200, 200), dtype=np.complex128)
    Shifted = np.zeros((200, 200), dtype=np.complex128)
    G = np.identity(200, dtype=np.complex128)
    Q = np.identity(200, dtype=np.complex128)
    D = np.diag(np.full(200, pilot))
    x_prev = np.zeros((200,), dtype=np.complex128)
    M_n = np.zeros((7,200), dtype=np.complex128)
    hat_hn = np.zeros((7, 200), dtype=np.complex128)
    reshaped_data_symbols = np.zeros((7, 200, 200), dtype=np.complex128)


    for m in range(2):
        coefficient = j0(2*np.pi*fd * m * T)
        for k in range(200):
            for l in range(200):
                # rklm = coefficient * (1-2j * np.pi * (l -k)*sigma_t/T)/(1+(4 * np.pi **2) * ((l-k)**2) * (sigma_t**2)/(T**2))
                if m == 1:
                    Sigma_prev[k,l] = coefficient
                elif m == 0:
                    Shifted[k,l] = coefficient

    A = np.dot(np.dot(Shifted**2, abs(Sigma_prev)**2), np.linalg.pinv(Sigma_prev))
    C = -A
    for t, symbols_at_T in enumerate(pilot_symbols):
        M_n = np.dot(np.dot(C, Sigma_prev), C.conjugate()) + np.dot(G, G.conjugate())
        lambda_ = np.dot(np.dot(D, M_n), D.conjugate()) + np.dot(variance_w**2, np.identity(200))
        K = np.dot(np.dot(M_n, D.conjugate()), np.linalg.inv(lambda_))
        xn = np.dot(C, x_prev) + np.dot(K, (symbols_at_T - np.dot(np.dot(D, C), x_prev)))
        Sigma_n = np.dot((np.identity(200) - np.dot(K, D)), M_n)
        x_prev = xn
        Sigma_prev = Sigma_n
        hat_hn[t] = np.dot(np.identity(200, dtype=np.complex128), xn) # Same as xn

    if plot:
        plot_filters(K, M_n, lambda_, Sigma_n, data_symbols, pilot_symbols, hat_hn)
    return hat_hn


def Kalman_filter_per_channel(pilot_symbols, variance_w, data_symbols, plot=True):
    pilot = 0.707 + 0.707j
    T, fd = (2048+160)/30720000, 50
    Sigma_n = np.zeros((7,200), dtype=np.complex128)
    M_n = np.zeros((7,200), dtype=np.complex128)
    auto_correlation = [j0(2*np.pi*fd * 0 * T), j0(2*np.pi*fd * 1 *T)] # By equation from paper

    Sigma_prev = np.full((200),2000*(auto_correlation[0]**2 - abs(auto_correlation[1])**2) / auto_correlation[0])
    print("Sigma_prev", Sigma_prev)
    lambda_ = np.zeros((7,200), dtype=np.complex128)
    K = np.zeros((7,200), dtype=np.complex128)
    xn = np.zeros((7,200), dtype=np.complex128)
    errors = np.zeros((7,200), dtype=np.complex128)
    a = auto_correlation[1] / auto_correlation[0]
    a = -a
    C = a

    # print("Sigma_prev", Sigma_prev)

    xn_prev = np.full((200) ,1)
    g = (auto_correlation[0]**2 - abs(auto_correlation[1])**2) / auto_correlation[0] # Not correct yet?

    for t, symbols_at_T in enumerate(pilot_symbols):

        M_n[t,:] = C * Sigma_prev * np.conjugate(C) + 1
        if M_n[t,:].shape != (200,):
            raise Exception(f"M_N at T: {M_n[t,:].shape}")

        lambda_[t,:] = pilot * M_n[t,:] * np.conjugate(pilot) + variance_w

        K[t,:] = M_n[t,:] * np.conjugate(pilot) / lambda_[t,:]

        if symbols_at_T.shape != (200,):
            raise Exception(f"Symbols at T: {symbols_at_T.shape}")
        errors[t,:] = (symbols_at_T - pilot*C* xn_prev)
        xn[t,:] = C*xn_prev + K[t,:] * errors[t,:]
        errors[t,:] = errors[t,:] / xn[t,:] # Normalize
        Sigma_n[t,:] = (1- K[t,:] * pilot) * M_n[t,:]
        xn_prev = xn[t,:]
        Sigma_prev = Sigma_n[t,:]


    # print("M_n", M_n[0,:])
    # print("lambda_", lambda_[0,:])
    # print("K", K[0,:])
    # print("xn", xn[0,:])
    # print("Sigma_n", Sigma_n[0,:])
    xn = linear_combiner(xn)
    if plot:
        plot_filters(K, M_n, lambda_, Sigma_n, data_symbols, pilot_symbols, xn,  kalman='per_channel', errors=errors)
    return xn

def plot_filters(K, M_n, lambda_, Sigma_n, data_symbols, pilot_symbols, hn, kalman='total', errors=None):
    fig_1, axs_1 = plt.subplots(5, figsize=(10, 8))
    im0 = axs_1[0].imshow(abs(K), aspect='auto')
    im1 = axs_1[1].imshow(abs(M_n), aspect='auto')
    im2 = axs_1[2].imshow(abs(lambda_), aspect='auto')
    im3 = axs_1[3].imshow(abs(Sigma_n), aspect='auto')
    im4 = axs_1[4].imshow(abs(errors), aspect='auto')

    axs_1[0].set_title('Kalman filter coefficients')
    axs_1[1].set_title('M_n')
    axs_1[2].set_title('lambda')
    axs_1[3].set_title('Sigma_n')
    axs_1[4].set_title('Errors')
    fig_1.colorbar(im0, ax=axs_1[0], shrink=0.8)
    fig_1.colorbar(im1, ax=axs_1[1], shrink=0.8)
    fig_1.colorbar(im2, ax=axs_1[2], shrink=0.8)
    fig_1.colorbar(im3, ax=axs_1[3], shrink=0.8)
    fig_1.colorbar(im4, ax=axs_1[4], shrink=0.8)

    fig_2, axs_2 = plt.subplots(3,3, figsize=(10, 8))

    if kalman == 'per_channel':
        im4 = axs_2[0, 0].imshow(abs(data_symbols[:,::5]/hn[:]), aspect='auto')
        im5 = axs_2[1, 0].imshow(np.angle(data_symbols[:,::5]/hn[ :]))
    else:
        im4 = axs_2[0, 0].imshow(abs(np.divide(data_symbols[:,::5],hn)))
        im5 = axs_2[1, 0].imshow(np.angle(np.divide(data_symbols[:,::5],hn)))
    im6 = axs_2[2, 0].imshow(abs(hn), aspect='auto')
    im7 = axs_2[0, 1].imshow(np.angle(hn), aspect='auto')
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

def linear_combiner(hn):
    print("shape hn: ", hn.shape)


    # Take a moving average over the last axis of the array
    # hn = np.concatenate((hn, np.tile(hn[:, -1:], (1, window_size/2-1))), axis=1)
    cutoff = 0.98
    window_size = 11

    # Create the lowpass filter
    lowpass = signal.firwin(window_size, cutoff)

    hn_pad = np.concatenate((np.tile(hn[:, :1], (1, window_size//2)), hn, np.tile(hn[:, -1:], (1, window_size//2))), axis=1)
    print("shape hn: ", hn_pad.shape)
    hn_filtered = np.apply_along_axis(lambda x: np.convolve(x, lowpass, mode='valid'), axis=-1, arr=hn_pad)

    print(hn_filtered.shape)
    
    # return hn
    return hn_filtered