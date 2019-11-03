import numpy as np
import matplotlib.pylab as plt
import skimage as ski
import skimage.io

mag = np.zeros((100,100))  # magnitude (all zeros for now)
ph = np.zeros((100,100))   # phase (all zeros for now)

mag[1,2] = 1          # 1 cycle in x, 2 cycles in y

y = mag*np.exp(1j*ph)   # build fft combining mag and phase
x = np.real(np.fft.ifft2(y))    # inverse fft (then take real part)
plt.imshow(x)           # plot
plt.show()


plt.plot(x[:,0])
plt.plot(x[0,:], 'r')
plt.legend(['x axis','y axis'])
plt.show()


im = ski.io.imread('cat.jpg')  # read image
im = im[:,:,1]  # only keep one colour channel

plt.imshow(im)
plt.show()

y = np.fft.fft2(im)

clim = np.quantile(np.abs(y.reshape(-1)), [.01, .99])
plt.imshow(np.fft.fftshift(np.abs(y)), vmin=clim[0], vmax=clim[1])
plt.gray()
plt.title('magnitude')
plt.show()

clim = np.quantile(np.angle(y.reshape(-1)), [.01, .99])
plt.imshow(np.fft.fftshift(np.angle(y)), vmin=clim[0], vmax=clim[1])
plt.title('phase')
plt.show()


mag = np.abs(y)
x   = np.fft.ifft2(mag)
plt.imshow(np.real(x))
plt.show()

clim = np.quantile(np.real(x.reshape(-1)), [0.01, 0.99])
plt.imshow(np.real(x), vmin=clim[0], vmax=clim[1])
plt.show()


ph = np.angle(y)
x   = np.fft.ifft2(np.exp(1j*ph))
clim = np.quantile(np.real(x.reshape(-1)), [0.01, 0.99])
plt.imshow(np.real(x), vmin=clim[0], vmax=clim[1])
plt.show()


siz = [5, 5]  # size of the box
box = np.ones_like(im)
box[200-siz[0]:200+siz[0],300-siz[1]:300+siz[1]] = 0
plt.imshow(box)
plt.show()


y = np.fft.fftshift(np.fft.fft2(im))
y = y*box
x = np.fft.ifft2(y)
plt.imshow(np.real(x))
plt.show()


box = np.logical_not(box).astype(int) # this turns zeros to ones and ones to zeros
y = np.fft.fftshift(np.fft.fft2(im))
y = y*box
x = np.fft.ifft2(y)
plt.imshow(np.real(x))
plt.show()

