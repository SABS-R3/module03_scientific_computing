import numpy as np
import scipy
import matplotlib.pylab as plt

# mean position
mu = np.array([5,2])

# angle in degrees
ang = 60;

# rotation matrix
ang_rad = np.radians(ang)
R = np.array([
        [np.cos(ang_rad), -np.sin(ang_rad)],
        [np.sin(ang_rad),  np.cos(ang_rad)],
])

# rotated covariance matrix
sig = R @ np.array([[1.3, 0], [0, .1]]) @ R.transpose()

# create points using multivariate gaussian random
y = np.random.multivariate_normal(mu, sig, 100)

# plot the data
plt.plot(y[: , 0], y[: , 1], 'b.')
plt.xlabel('x')
plt.ylabel('y')

ydemean = y - np.mean(y,axis=0)
plt.plot(ydemean[:,0], ydemean[:,1], 'k.')
plt.show()

d,v = np.linalg.eig(np.cov(y.transpose()))
v = np.abs(v)
d = np.sqrt(np.abs(d))


# Plot the two eigenvectors
plt.plot(ydemean[:,0], ydemean[:,1], 'k.')
plt.arrow(0,0,d[0]*v[0,0],d[0]*v[1,0],color='b',width=0.05)   # minor eigenvector
plt.arrow(0,0,d[1]*v[0,1],d[1]*v[1,1],color='r',width=0.05)   # major eigenvector
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

yy = (ydemean @ v[:,1].reshape(-1,1)) @ v[:,1].reshape(1,-1)
plt.plot(yy[:,0], yy[:,1], 'g.')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


t = 2*np.pi*np.random.rand(1000,1) # random angle
r = .5+np.random.rand(1000,1)      # random radius between .5 and 1.5
x = r*np.sin(t)
y = r*np.cos(t);

plt.clf()
plt.plot(x,y,'o')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

Y = np.concatenate([x, y], axis=1)
d,v = np.linalg.eig(np.cov(Y.transpose()))
v = np.abs(v)
d = np.sqrt(np.abs(d))

plt.plot(x,y,'o')
plt.arrow(0,0,d[1]*v[0,1],d[1]*v[1,1],color='r',width=0.05)
plt.arrow(0,0,d[0]*v[0,0],d[0]*v[1,0],color='b',width=0.05)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


ydemean = Y - np.mean(Y, axis=0)
yy = (ydemean @ v[:,1].reshape(-1,1))*v[:,1].reshape(1,-1)
plt.plot(yy[:,0],yy[:,1],'g.');
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


y1 = np.random.multivariate_normal(np.array([0,0]),
                                   np.array([[1,0],[0,20]]),
                                   1000)
y2 = np.random.multivariate_normal(np.array([5,0]),
                                    np.array([[1,0],[0,20]]),
                                    1000);
y = np.concatenate([y1,y2],axis=0)
y = y - np.mean(y, axis=0)    # demeand the data to keep things simple

plt.clf()
plt.plot(y[:,0],y[:,1],'k.')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

d,v = np.linalg.eig(np.cov(y.transpose()))
v = np.abs(v)
d = np.sqrt(np.abs(d))
plt.plot(y[:,0],y[:,1],'k.')
plt.arrow(0,0,d[1]*v[0,1],d[1]*v[1,1],color='r',width=0.05); # major eigenvector
plt.arrow(0,0,d[0]*v[0,0],d[0]*v[1,0],color='b',width=0.05); # minor eigenvector
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

yy = (y @ v[:,1].reshape(-1,1)) @ v[:,1].reshape(1,-1);
plt.plot(yy[:,0],yy[:,1],'g.')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


