import scipy.io
import numpy as np
import matplotlib.pylab as plt

eeg = scipy.io.loadmat('eeg.mat')['eeg'].reshape(-1)

samp = 128                 # sampling rate in Hz
T    = 1/samp*len(eeg)     # total duration (in seconds)

N    = len(eeg)            # store the number of time points
f    = np.linspace(0,N-1,N)/T  # vector of frequencies for plotting

y = np.fft.fft(eeg)
plt.plot(f[:N//2],np.abs(y[:N//2]))
plt.show()


window = 1;            # 1 second window
nbins  = window*samp;  # corresponding number of time points
print(nbins)

F = np.zeros((nbins//2,N-nbins))
print(F.shape)

for i in range(N-nbins):
    x = eeg[i:i+nbins]
    y = np.fft.fft(x)
    F[:,i] = np.abs(y[:nbins//2])  # only keep half of the fft


plt.imshow(np.log1p(F), aspect='auto')
plt.yticks(ticks=np.arange(1,nbins/2,4), labels=np.arange(0,nbins/2-1,4))
plt.xlabel('time')
plt.show()

