import numpy as np
import matplotlib.pylab as plt

mag  = 2;     # magnitude (arbitrary units)
freq = 5;     # frequency in Hz
samp = 100;   # sampling rate in Hz

t = np.arange(0.0,1.0,1.0/samp)  # time (1s of data)
t = t[0:-1]                      # remove last time point
N = len(t)                       # store the number of time points

x = mag*np.cos(2*np.pi*freq*t)   # the signal equation
plt.plot(t,x,'.-')
plt.show()

y = np.fft.fft(x);                  # do Fast Fourier Transform
f = np.linspace(0.0,N-1.0,N);   # vector of frequencies for plotting

plt.plot(f,np.abs(y),'.-');      # plotting the magnitude of the FFT
plt.show()


plt.plot(f[:N//2],np.abs(y[:N//2])) # plotting half of the fft results
plt.xlabel('Freq (Hz)')           # labelling x-axis
plt.show()

print(np.sum(x**2))
print(np.mean(np.abs(y)**2))

x2 = x+3
y  = np.fft.fft(x2)
plt.plot(f[:N//2],np.abs(y[:N//2]),'.-')
plt.show()


phase = 2*np.pi*10
x2    = mag*np.cos(2*np.pi*freq*t+phase)
y = np.fft.fft(x2)
plt.plot(f[:N//2],np.abs(y[:N//2]),'.-')
plt.show()

mag = np.zeros(100)
mag[3] = 1        # spike in the magnitude at freq=3
mag[5] = .3       # spike in the magnitude at freq=5

ph = np.zeros(100) # zero phase throughout

y = mag*np.exp(1j*ph)  # the complex signal (fft of some real signal)
x = np.fft.ifft(y)         # the inverse fft (x is in general complex too)
x = np.real(x)         # here we imagine that we can "measure" the real part of x

z = np.fft.ifft(mag)       # here we just use the magnitude to do the ifft
z = np.real(z)

plt.plot(x,'.-r')   # plot the real part of the signal
plt.plot(z,'.-b')   # plot the ifft of the magnitude
plt.show()

mag = 2      # magnitude (arbitrary units)
dec = .3     # decay rate (in seconds)
samp = 100   # sampling rate in Hz

t = np.arange(0.0,1.0,1/samp)  # time (1s of data)
t = t[:-1]                     # remove last time point
N = len(t)                     # store the number of time points

x = mag*np.exp(-t/dec)         # the signal equation
plt.plot(t,x,'.-')
plt.show()

y = np.fft.fft(x)             # do Fast Fourier Transform
f = np.linspace(0,N-1,N)  # vector of frequencies for plotting

plt.plot(f[:N//2],np.abs(y[:N//2]),'.-')  # plot powerspectrum
plt.show()



mag = 2      # magnitude (arbitrary units)
t0  = .3     # peak location (in seconds)
samp = 100   # sampling rate in Hz

t = np.arange(0.0,1.0,1.0/samp)  # time (1s of data)
t = t[:-1]                       # remove last time point
N = len(t)                       # store the number of time points

x = np.zeros(N)              # signal
x[np.round(samp*t0).astype(int)] = mag

plt.plot(t,x,'.-');
plt.show()

y = np.fft.fft(x);             # do Fast Fourier Transform
f = np.linspace(0,N-1,N);  # vector of frequencies for plotting

plt.plot(f[:N//2],np.abs(y[:N//2]),'.-')            # plot powerspectrum
plt.show()

plt.plot(f[:N//2],np.unwrap(np.angle(y[:N//2])),'.-')  # plot unwrapped phase of FFT
plt.show()


x = np.random.randn(1000)
y = np.fft.fft(x)

plt.plot(x)
plt.show()
plt.plot(np.abs(y[:500]))
plt.show()


x = np.random.randn(1000)

span = 10  # try changing this
x = np.convolve(x, np.ones((span,))/span, mode='valid')
y = np.fft.fft(x)
plt.plot(x)
plt.show()
plt.plot(np.abs(y[:500]))
plt.show()


