import numpy as np
import matplotlib.pyplot as plt


def model(x):
    return -2*x**3 + 3*x**2 + x + 1


def generate_data():
    n = 100
    x = np.linspace(0, 2, n)
    y = model(x) + 1.0*np.random.randn(n)
    return x, y


def fit(m, x, y, eta):
    print('fit({})'.format(m))
    n = len(x)
    print('m = ',m)
    print('n = ',n)

    design = np.empty((n, m))
    for i in range(m):
        design[:, i] = x**i

    beta = np.linalg.solve(
            np.matmul(design.transpose(), design) + eta*np.eye(m, m),
            np.matmul(design.transpose(), y)
            )

    fit_n = 1000
    fit_x = np.linspace(x[0], x[-1], fit_n)
    fit_design = np.empty((fit_n,m))
    for i in range(m):
        fit_design[:, i] = fit_x**i

    fit_y = np.matmul(fit_design, beta)

    plt.clf()
    plt.plot(x, y, '.', label='data')
    plt.plot(fit_x, fit_y, label='fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('m = {}'.format(m))
    plt.legend()
    plt.savefig('fit_{}.pdf'.format(m))

    residual = np.linalg.norm(np.matmul(design, beta) - y)

    loo_error = 0.0
    for l in range(n):
        loo_x = np.delete(x, l)
        loo_y = np.delete(y, l)
        loo_design = np.empty((n-1, m))
        for i in range(m):
            loo_design[:, i] = loo_x**i

        loo_beta = np.linalg.solve(
            np.matmul(loo_design.transpose(), loo_design) + eta*np.eye(m, m),
            np.matmul(loo_design.transpose(), loo_y)
            )

        loo_design = np.empty((1, m))
        for i in range(m):
            loo_design[:, i] = x[l]**i

        loo_error += (np.matmul(loo_design, loo_beta)[0] - y[l])**2

    loocv = np.sqrt(loo_error)

    return residual, loocv

def plot_loocv_and_residual():
    ms = np.arange(1,40)
    loocv_low = np.empty(len(ms))
    residual_low = np.empty(len(ms))
    loocv_high = np.empty(len(ms))
    residual_high = np.empty(len(ms))

    x, y = generate_data()
    for i, m in enumerate(ms):
        residual_low[i], loocv_low[i] = fit(m, x, y, 1e-7)
        residual_high[i], loocv_high[i] = fit(m, x, y, 1e-3)

    plt.clf()
    plt.plot(ms, residual_low, label='low')
    plt.plot(ms, residual_high, label='low')
    plt.legend()
    plt.xlabel('m')
    plt.ylabel('residual')
    plt.savefig('residual.pdf')

    plt.clf()
    plt.plot(ms, loocv_low, label='low')
    plt.plot(ms, loocv_high, label='high')
    plt.legend()
    plt.xlabel('m')
    plt.ylabel('loocv')
    plt.savefig('loocv.pdf')

    min_i = np.argmin(loocv_low)
    print('minimum loocv of {} achieved for m = {}'.format(loocv_low[min_i], ms[min_i]))


if __name__ == '__main__':
    plot_loocv_and_residual()
