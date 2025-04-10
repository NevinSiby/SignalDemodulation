import numpy as np
import matplotlib.pyplot as mplt
import scipy.io.wavfile as wv
import scipy.signal as sgnl
from scipy.io.wavfile import write

samplerate, data = wv.read(https://github.com/NevinSiby/SignalDemodulation/blob/main/modulated_noisy_audio.wav)


fft_data = np.fft.fft(data)
freqs = np.fft.fftfreq(len(data), d = 1/samplerate)


mag = np.abs(fft_data[:len(fft_data)//2])
d=np.argpartition(mag,-2)
f1, f2 = d[-2:]
fc = (freqs[:len(freqs)//2][f1]+freqs[:len(freqs)//2][f2])/2





mplt.plot(freqs[:len(freqs)//2], np.abs(fft_data)[:len(fft_data)//2])
mplt.title("Fast Fourier Transformation")
mplt.xlabel("Frequency")
mplt.ylabel("Amplitude")
mplt.savefig("FFT_plot of modulated signal.png")


t=np.arange(len(data))/samplerate 
demod = data*np.cos(2*np.pi*fc*t)


cutoff = 4000 
order =5
nyquist = samplerate / 2
norm_cutoff = cutoff / nyquist
b, a = sgnl.butter(order, norm_cutoff, btype='low')
recovered = sgnl.filtfilt(b, a, demod)



new_fft = np.fft.fft(recovered)
new_freqs = np.fft.fftfreq(len(recovered), d=(1/samplerate))


mplt.plot(new_freqs[:len(new_freqs)//2], np.abs(new_fft[:len(new_fft)//2]))
mplt.title("fft of recovered signal")
mplt.xlabel("freq")
mplt.ylabel("amp")
mplt.savefig("fplot of fft of recovered signal.png")




recovered_normalized = recovered / np.max(np.abs(recovered))
recovered_int16 = np.int16(recovered_normalized * 32767)

write("recovered_audio.wav", samplerate, recovered_int16)
