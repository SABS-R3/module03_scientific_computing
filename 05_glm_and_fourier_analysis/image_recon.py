import numpy as np
import matplotlib.pylab as plt
import skimage as ski
import skimage.transform

im = ski.data.shepp_logan_phantom()
plt.imshow(im)
plt.gray()
plt.show()

R = ski.transform.radon(im, [0])
plt.plot(R)
plt.show()


R = ski.transform.radon(im, np.linspace(0,180,40))   # 40 angles from 0 to 180 degrees
plt.imshow(R)
plt.show()


x = ski.transform.iradon(R, np.linspace(0,180,40))
plt.imshow(x)
plt.show()


im[100,80] = 100

R = ski.transform.radon(im, np.linspace(0,180,100))
x = ski.transform.iradon(R, np.linspace(0,180,100))
plt.imshow(x, vmin=0, vmax=1)
plt.show()


im = ski.data.shepp_logan_phantom()
proj = np.sum(im,axis=0)   # projection on the x axis (along the y axis)
f1d  = np.fft.fft(proj)    # 1D Fourier
f2d  = np.fft.fft2(im)     # 2D Fourier
xslice = f2d[0,:]   # slice along x (through zero frequency)

plt.plot(np.abs(f1d),'b')
plt.plot(np.abs(xslice),'r--')
plt.show()
plt.plot(np.angle(f1d),'b')
plt.plot(np.angle(xslice),'r--')
plt.show()

im = ski.io.imread('brain.bmp')

data = np.fft.fftshift(np.fft.fft2(im))

K = data[:,112:112+32]   # this turns k-space from 256x256 to 256x32


x = np.fft.ifft2(K,s=[256,256])
plt.imshow(np.abs(x))    # we typically look at the magnitude
plt.show()


K = data
K[::2,:] = 0.8*K[::2,:]
x = np.fft.ifft2(K)
plt.imshow(np.abs(x))
plt.show()


K = data
K[:,::2] = 0.8*K[:,::2]
x = np.fft.ifft2(K)
plt.imshow(np.abs(x))
plt.show()


T2star = 20            # milliseconds
Ttotal = 100           # time it takes to "read" k-space
N      = data.shape[0] # number of pixels

K = data
for i in range(K.shape[1]):
    K[:,i] = K[:,i]*np.exp(-np.linspace(0,Ttotal,N)/T2star)


x = np.fft.ifft2(K)
plt.imshow(np.abs(x))
plt.show()

