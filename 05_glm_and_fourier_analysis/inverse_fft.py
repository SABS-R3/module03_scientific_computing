import numpy as np
import matplotlib.pylab as plt

mag  = 2;     # magnitude (arbitrary units)
freq = 5;     # frequency in Hz
samp = 100;   # sampling rate in Hz

t = np.arange(0,1,1/samp)  # time (1s of data)
t = t[:-1]                 # remove last time point
N = len(t)                 # store the number of time points

x = mag*np.cos(2*np.pi*freq*t + 10)    # the signal equation (+phase)
plt.plot(t,x,'.-')
plt.show()

y  = np.fft.fft(x)           # do Fast Fourier Transform
z  = np.fft.ifft(y)

plt.plot(t,x,'.-')
plt.plot(t,z,'r--')
plt.show()



y2 = np.abs(y)
z  = np.fft.ifft(y2)

plt.plot(t,x)
plt.plot(t,z,'r--')
plt.show()


y2 = np.exp(1j*np.angle(y))
z  = np.fft.ifft(y2)
plt.plot(t,x)
plt.plot(t,z,'r--')
plt.show()


mag = np.abs(y)
mag[20] = 10   # spike in magnitude
ph = np.angle(y)
ph[10]  = 1    # spike in phase

y2 = mag*np.exp(1j*ph)
z  = np.real(np.fft.ifft(y2))
plt.plot(t,x)
plt.plot(t,z,'r--')
plt.show()
