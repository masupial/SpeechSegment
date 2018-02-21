from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ReadData import read_wav_file
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
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

def identify_spectrogram_clusters_1p2(data,sample_rate,clusters=2):
    '''
    trying other clustering on the the vertical lines again
    '''

    # hacked the plt.specgram to get the specrogram
    Pxx, freqs, bins, im = plt.specgram(data, Fs=sample_rate)

    # split the Pxx into two clusters
    # clusterModel = SpectralClustering(n_clusters=2, random_state=0).fit(np.transpose(Pxx))

    clusterModel = AgglomerativeClustering(n_clusters=2).fit(np.transpose(Pxx))

    return clusterModel.labels_,bins

def identify_spectrogram_clusters_2(data,sample_rate,clusters=2):
    '''
    trying k means clustering of derived features
    derived from : low dim version of the vertical lines
    '''

    # hacked the plt.specgram to get the spectrogram
    Pxx, freqs, bins, im = plt.specgram(data, Fs=sample_rate)

    # extract Pxx into some lower dimension
    # let's use various norms as features
    # np.ones((spectral_energy_feature.shape[0]))
    spectral_energy_feature = np.sum(Pxx**2, axis=0) # 2-norm
    #hack: add an extra meaningless dimension to this to have a smooth run
    spectral_sum_feature = np.sum(Pxx, axis=0) # 1-norm
    spectral_max_feature = np.max(Pxx, axis=0) # infinity norm
    spectral_feature = np.stack((spectral_sum_feature,spectral_energy_feature,spectral_max_feature),axis=1)

    # pca?, maybe later!
    #ipca = IncrementalPCA(n_components=2, batch_size=3)
    #ipca.fit(Pxx)
    #Pxx_transform = ipca.transform(Pxx)

    # split the Pxx into two clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(spectral_feature)

    return kmeans.labels_,bins


if __name__ == '__main__':
    sample_rate, data = read_wav_file('harvard_data/OSR_us_000_0010_8k.wav')
    number_samples = data.shape[0]

    # on the first few samples
    data= data[1:np.int(number_samples / 5)]

    # plotting the spectrogram
    ax_plot = plot_data_with_spectrogram(data,sample_rate)

    # identifying the clusters
    labels, bins = identify_spectrogram_clusters_2(data, sample_rate)
    # print(labels)
    for ii in range(labels.shape[0]):
        if labels[ii] == 1:
            plt.scatter(bins[ii],0, s=3, color="r")

    plt.show()


