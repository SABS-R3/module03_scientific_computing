import numpy as np
import scipy
import imageio
import matplotlib.pylab as plt

im = imageio.imread('world-map.gif').astype('float64')  # read image
im = np.sum(im, axis=2)                  # sum across the three colour channels
plt.imshow(im)                           # display the image
plt.gray()
plt.show()

d,v = np.linalg.eig(np.cov(im.transpose()))

x = im
m = np.mean(x, axis=1)

for i in range(x.shape[0]):
    x[i,:] = x[i,:] - m[i]

p = 4;

# reconstruct the data with top p eigenvectors
y = (x @ v[:,:p]) @ v[:,:p].transpose()

# plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.imshow(im)
ax2.imshow(np.real(y))

# add mean
for i in range(x.shape[0]):
    y[i,:] = y[i,:] + m[i]
ax3.imshow(np.abs(y))
plt.show()


im = imageio.imread('Queen_Victoria_by_Bassano.jpg').astype('float64')
im = np.sum(im, axis=2)

# reconstruct the data with top p eigenvectors
d,v = np.linalg.eig(np.cov(im.transpose()))
idx = d.argsort()[::-1]
v = v[:,idx]
p = 4
x = im
m = np.mean(x, axis=1)
for i in range(x.shape[0]):
    x[i,:] = x[i,:] - m[i]
y = (x @ v[:,:p]) @ v[:,:p].transpose()

# plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.imshow(im)
ax2.imshow(y)

# add mean
for i in range(x.shape[0]):
    y[i,:] = y[i,:] + m[i]
ax3.imshow(y)
plt.show()
