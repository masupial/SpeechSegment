from scipy.io import wavfile
import matplotlib.pyplot as plt

def read_wav_file(file_name):
    '''
    function to read the wav file
    '''

    sample_rate, data = wavfile.read(file_name)

    # optional code for plotting of the signal
    plt.figure(1)
    plt.title('speech wave')
    plt.plot(data)
    plt.show()


    return sample_rate, data


if __name__ == '__main__':
    sample_rate, data = read_wav_file('harvard_data/OSR_us_000_0010_8k.wav')
    print('sample_rate : ',sample_rate,' and number of samples : ',data.shape[0])