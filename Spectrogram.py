from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ReadData import read_wav_file
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

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
    # plt.show()

    # return the axis for later use by parent!
    return ax1

def identify_spectrogram_clusters_1(data,sample_rate,clusters=2):
    '''
    trying k means clustering of the the vertical lines directly
    '''

    # hacked the plt.specgram to get the specrogram
    Pxx, freqs, bins, im = plt.specgram(data, Fs=sample_rate)

    # split the Pxx into two clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.transpose(Pxx))

    return kmeans.labels_,bins

def identify_spectrogram_clusters_2(data,sample_rate,clusters=2):
    '''
    trying k means clustering of derived features
    derived from : extracted version of the vertical lines
    '''

    # hacked the plt.specgram to get the spectrogram
    Pxx, freqs, bins, im = plt.specgram(data, Fs=sample_rate)

    # extract Pxx into some lower dimension
    # first, let's sum of squares
    spectral_energy_feature = np.sum(Pxx**2, axis=0)
    #hack: add an extra meaningless dimension to this to have a smooth run
    spectral_energy_feature = np.stack((np.ones((spectral_energy_feature.shape[0])),spectral_energy_feature),axis=1)

    # pca?, maybe later!
    #ipca = IncrementalPCA(n_components=2, batch_size=3)
    #ipca.fit(Pxx)
    #Pxx_transform = ipca.transform(Pxx)

    # split the Pxx into two clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(spectral_energy_feature)

    return kmeans.labels_,bins


if __name__ == '__main__':
    sample_rate, data = read_wav_file('harvard_data/OSR_us_000_0010_8k.wav')
    number_samples = data.shape[0]

    # plotting the spectrogram on the first few samples
    ax_plot = plot_data_with_spectrogram(data[1:np.int(number_samples/5)],sample_rate)

    # identifying the clusters
    labels, bins = identify_spectrogram_clusters_2(data[1:np.int(number_samples / 5)], sample_rate)
    # print(labels)
    for ii in range(labels.shape[0]):
        if labels[ii] == 1:
            plt.scatter(bins[ii],0, s=3, color="r")

    plt.show()


