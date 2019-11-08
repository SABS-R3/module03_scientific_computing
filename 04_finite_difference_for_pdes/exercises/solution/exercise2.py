import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

def construct_laplace_matrix_1d(N, h):
    e = np.ones(N)
    diagonals = [e, -2*e, e]
    offsets = [-1, 0, 1]
    L = scipy.sparse.spdiags(diagonals, offsets, N, N) / h**2
    return L

def solve_fisher():
    eps = 0.0051
    h = 0.05
    k = 0.4 * h**2 / eps
    print(k)
    Tf = 10.0
    x = np.arange(0+h, 20-h, h)

    N = len(x)
    L = construct_laplace_matrix_1d(N, h)
    bc = np.concatenate(([1], np.zeros(N-1)))/h**2

    u = (x<3).astype('float64')

    fig, ax = plt.subplots()
    ln, = plt.plot(x, u)
    ax.set_ylim(0, 1.1)

    def update(frame):
        u_new = u + k*( eps*(L*u + bc) + (u-u**2) )
        u[:] = u_new

        ln.set_data(x, u)
        ax.set_title('t = {}'.format(frame*k))
        print(frame)
        return ln,

    numsteps = int(np.ceil(Tf/k))
    Tf = numsteps*k
    ani = FuncAnimation(fig, update, frames=numsteps, interval=30, blit=False,
            repeat=False)
    plt.show()

def solve_biharmonic():
    h = .025
    x = np.arange(-1+h, 1-h, h)
    N = len(x);
    u = (np.abs(x)<0.3).astype('float64');
    k = 0.05*h

    L = construct_laplace_matrix_1d(N, h).tolil()
    L[0,-1] = 1.0/h**2
    L[-1,0] = 1.0/h**2
    H = -L.dot(L);

    I = scipy.sparse.eye(N)
    A = I - k*H

    fig, ax = plt.subplots()
    ln, = plt.plot(x, u)
    ax.set_ylim(0, 1.1)

    def update(frame):
        #unew = u + k*(H*u);     %forward euler
        u_new = scipy.sparse.linalg.spsolve(A, u)
        u[:] = u_new
        ln.set_data(x, u)
        return ln,

    Tf = 0.5
    numsteps = np.ceil(Tf / k)
    Tf = k*numsteps

    ani = FuncAnimation(fig, update, frames=np.arange(0, Tf, k), interval=200, blit=True)
    plt.show()


def solve_kuramoto_sivashinsky():
    h = 0.025
    x = np.arange(-1+h, 1-h, h)
    N = len(x);
    u = (np.abs(x)<0.3).astype('float64');
    k = 0.05*h

    L = construct_laplace_matrix_1d(N, h).tolil()

    h = 32*np.pi/400;
    x = np.arange(-16*np.pi+h, 16*np.pi-h, h)
    N = len(x)
    u = np.cos(x/16)*(1+np.sin(x/16))

    k = 0.2
    L = construct_laplace_matrix_1d(N, h).tolil()
    L[0,-1] = 1.0/h**2
    L[-1,0] = 1.0/h**2
    H = L.dot(L)

    e = np.ones(N)
    D = scipy.sparse.spdiags([-e, e], [-1, 1], N, N)
    D = 1/(2*h) * D

    I = scipy.sparse.eye(N)

    A = I + k*H + k*L

    fig, ax = plt.subplots()
    ln, = plt.plot(x, u)
    ax.set_ylim(-4, 4)

    def update(frame):
        b = u - k*(D*(u**2/2))
        u_new = scipy.sparse.linalg.spsolve(A, b)
        u[:] = u_new
        ln.set_data(x, u)
        return ln,

    Tf = 0.5
    numsteps = np.ceil(Tf / k)
    Tf = k*numsteps

    ani = FuncAnimation(fig, update, frames=np.arange(0, Tf, k), interval=30, blit=True)
    plt.show()


if __name__ == '__main__':
    solve_fisher()

