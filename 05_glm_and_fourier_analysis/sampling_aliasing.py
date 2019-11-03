import numpy as np
import matplotlib.pylab as plt
import skimage as ski
import skimage.io


freq1 = 5.0      # freq in Hz
freq2 = 1.0      # freq in Hz
samp  = 1000.0   # sampling rate in Hz

t = np.arange(0,1,1/samp)  # time (1s of data)
t = t[0:-1]                # remove last time point
N = len(t)                 # store the number of time points

x = np.cos(2*np.pi*freq1*t) + .5*np.cos(2*np.pi*freq2*t)    # the signal equation
plt.plot(t,x,'.-')
plt.show()



F = 50.0         # sub-sampling frequency (Hz)
T = 1/F          # sub-sampling period (sec)
n = int(N*T)     # sub-sampling period (in samples)

z = x[::n]         # sub-sampled signal
y = np.fft.fft(z)  # fft of sub-sampled signal

subN = len(z)  # length of sub-sampled signal


f = np.linspace(0,subN-1,subN)  # vector of frequencies for plotting

plt.plot(f[:subN//2], np.abs(y[:subN//2])) # plot powerspectrum as before
plt.show()


freq1 = 5.0      # freq in Hz
freq2 = 1.0      # freq in Hz
samp  = 1000.0   # sampling rate in Hz

t = np.arange(0,1,1/samp)  # time (1s of data)
t = t[:-1]                 # remove last time point
N = len(t)                 # store the number of time points

x = np.cos(2*np.pi*freq1*t) + .5*np.cos(2*np.pi*freq2*t)    # the signal equation
plt.plot(t,x,'.-')


F = 50.0         # sub-sampling frequency (Hz)
T = 1/F          # sub-sampling period (sec)
n = int(N*T)     # sub-sampling perior (in samples)

z = x[::n]  # sub-sampled signal
t = t[::n]
y = np.fft.fft(z)


w = np.fft.ifft(y)
plt.plot(t,w,'r--')
plt.show()


im = ski.io.imread('cat.jpg')
x = im[::2,::2,0]
plt.imshow(x)
plt.show()


y = np.fft.fft2(x)
clim = np.quantile(np.abs(y.reshape(-1)),[.01, .99])
plt.imshow(np.fft.fftshift(np.abs(y)), vmin=clim[0], vmax=clim[1])
plt.gray()
plt.show()


y = np.fft.fftshift(y) # we need this before zero padding
z = np.fft.ifft2(y, s=x.shape) # zero padded ifft
clim = np.quantile(np.abs(z.reshape(-1)), [.01, .99])
plt.imshow(np.abs(z), vmin=clim[0], vmax=clim[1])
plt.show()

siz = [3, 3]   # 3x3 box. You can play with changing this
box = np.ones(siz)/9

x = im[:,:,0]
y = np.fft.fft2(x)

f = np.fft.fft2(box, s=x.shape)  # use zeropadding

z = np.fft.ifft2(y*f)  # inverse FFT of the product

clim = np.quantile(np.abs(z.reshape(-1)), [.01, .99])
plt.imshow(np.abs(z), vmin=clim[0], vmax=clim[1])


gauss = np.zeros((15,15))   # size of the filter can be changed

sigma  = 10             # sigma of the Gaussian (in pixels)
centre = np.array(gauss.shape)//2 # centre of the box

for i in range(gauss.shape[0]):
    for j in range(gauss.shape[1]):
        gauss[i,j] = np.exp(- np.sum(([i, j]-centre)**2)/2/sigma**2 )
gauss = gauss/np.sum(gauss)

plt.imshow(gauss)
plt.show()


x = im[:,:,0]
y = np.fft.fft2(x)

f = np.fft.fft2(gauss, s=x.shape)  # use zeropadding

z = np.fft.ifft2(y*f)  # inverse FFT of the product

clim = np.quantile(np.abs(z.reshape(-1)), [.01, .99])
plt.imshow(np.abs(z), vmin=clim[0], vmax=clim[1])
plt.show()


box = [[-1, 1]]    # along x-dimension
box = [[-1],
       [ 1]]    # along y-dimension
box = [[-1, 0, 1]]  # a slightly better filter along x
box = [[-1],
       [ 0],
       [ 1]]  # a slightly better filter along y

x = im[:,:,0]
y = np.fft.fft2(x)

box = [[-1, 0, 1]] # try with different ones if you have time
f = np.fft.fft2(box, s=x.shape)  # use zeropadding

z = np.fft.ifft2(y*f)  # inverse FFT of the product

clim = np.quantile(np.abs(z.reshape(-1)), [.01, .99])
plt.imshow(np.abs(z), vmin=clim[0], vmax=clim[1])
plt.show()

