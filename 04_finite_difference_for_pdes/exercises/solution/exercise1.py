import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time

def construct_laplace_matrix_1d(N, h):
    e = np.ones(N)
    diagonals = [e, -2*e, e]
    offsets = [-1, 0, 1]
    L = scipy.sparse.spdiags(diagonals, offsets, N, N) / h**2
    return L

def construct_laplace_matrix_2d(N, h):
    L = construct_laplace_matrix_1d(N, h)
    I = scipy.sparse.eye(N)
    return scipy.sparse.kron(L, I) + scipy.sparse.kron(I, L)

def solve_poisson_dirichelet():
    N = 100
    x = np.linspace(0, 1, N)
    h = x[1]-x[0]
    x_interior = x[1:-1]
    L = construct_laplace_matrix_1d(N-2, h)
    b = -np.ones(N-2)
    y_interior = np.linalg.solve(L.toarray(), b)
    y = np.concatenate(([0], y_interior, [0]))
    y_exact = 0.5*x - 0.5*x**2

    plt.clf()
    plt.plot(x, y, label='FD')
    plt.plot(x, y_exact, label='exact')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def solve_poisson_neumann():
    N = 100
    x = np.linspace(0, 1, N)
    h = x[1]-x[0]
    x_interior = x[1:-1]
    L = construct_laplace_matrix_1d(N-2, h).toarray()
    L[-1,-1] /= 2.0
    b = -np.ones(N-2)
    y_interior = np.linalg.solve(L, b)
    y = np.concatenate(([0], y_interior, [y_interior[-1]]))
    y_exact = x - 0.5*x**2

    plt.clf()
    plt.plot(x, y, label='FD')
    plt.plot(x, y_exact, label='exact')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def solve_laplace_2d():
    N = 100
    x = np.linspace(-1, 1, N)
    h = x[1]-x[0]
    [xx, yy] = np.meshgrid(x[1:-1], x[1:-1])
    x = xx.reshape(-1)
    y = yy.reshape(-1)
    L = construct_laplace_matrix_2d(N-2, h)
    b = -np.ones_like(x)
    u = scipy.sparse.linalg.spsolve(L, b)
    u_exact = 0.5 - 0.5*x**2

    for k in range(1,100,2):
        u_exact -= 16.0/np.pi**3*np.sin(k*np.pi*(1+x)/2)/(k**3*np.sinh(k*np.pi))*(np.sinh(k*np.pi*(1+y)/2)+np.sinh(k*np.pi*(1-y)/2))

    error = np.linalg.norm(u_exact - u)/np.linalg.norm(u_exact)

    print('error is {}'.format(error))

    plt.clf()
    plt.imshow(u.reshape((N-2, N-2)))
    plt.show()



def compare_dense_and_sparse():
    Ns = (10**np.linspace(1, 3, 100)).astype('int')
    time_sparse = np.empty_like(Ns, dtype='float64')
    time_dense = np.empty_like(Ns, dtype='float64')
    for i,N in enumerate(Ns):
        print(N)
        x = np.linspace(0, 1, N)
        h = x[1]-x[0]
        x_interior = x[1:-1]
        L = construct_laplace_matrix_1d(N-2, h).tocsr()
        b = -np.ones(N-2)
        L_dense = L.toarray()
        t0 = time.perf_counter()
        y_interior_sparse = scipy.sparse.linalg.spsolve(L, b)
        t1 = time.perf_counter()
        time_sparse[i] = t1-t0
        t0 = time.perf_counter()
        y_interior_dense = scipy.linalg.solve(L_dense, b)
        t1 = time.perf_counter()
        time_dense[i] = t1-t0



    plt.clf()
    plt.loglog(Ns, time_dense, label='dense')
    plt.loglog(Ns, time_sparse, label='sparse')
    plt.xlabel('N')
    plt.ylabel('time')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    compare_dense_and_sparse()

