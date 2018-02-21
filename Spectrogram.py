from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ReadData import read_wav_file

def plot_data_with_spectrogram(data,sample_rate):
    '''
    plot the time series and spectrogram in a single plot
    '''

    dt = 1 / sample_rate
    number_samples = data.shape[0]
    t = np.arange(0.0,  number_samples* dt, dt)

    # time data first
    ax1 = plt.subplot(211)
    plt.ylabel('speech wave')
    plt.plot(t, data)

    # now plot spectrogram
    plt.subplot(212, sharex=ax1)
    plt.ylabel('spectrogram')
    Pxx, freqs, bins, im = plt.specgram(data, Fs=sample_rate)
    # plt.pcolormesh(bins, freqs, Pxx)
    plt.show()

if __name__ == '__main__':
    sample_rate, data = read_wav_file('harvard_data/OSR_us_000_0010_8k.wav')
    number_samples = data.shape[0]

    # plotting the spectrogram on the first few samples
    plot_data_with_spectrogram(data[1:np.int(number_samples/5)],sample_rate)

