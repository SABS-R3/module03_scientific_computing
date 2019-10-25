from scipy.integrate import odeint
from scipy.optimize import minimize, shgo
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def calculate_jac():
    y = sp.Symbol('y')
    t = sp.Symbol('t')
    r = sp.Symbol('r')
    k = sp.Symbol('k')
    p0 = sp.Symbol('p0')
    eq = logistic_rhs(y, t, r, k)
    sp.pprint(eq)
    sp.pprint(eq.diff(y))
    sp.pprint(eq.diff(r))
    sp.pprint(eq.diff(k))

    eq = k / (1 + (k/p0 - 1)*sp.exp(-r*t))
    sp.pprint(eq)
    sp.pprint(eq.diff(y))
    sp.pprint(eq.diff(r))
    sp.pprint(eq.diff(k))


def logistic_rhs(y, t, r, k):
    return r*y*(1.0 - y/k)

def logistic_dy(y, t, r, k):
    return r - 2*r*y/k

def logistic_dp(y, t, r, k):
    return np.array([y*(1.0 - y/k), r*y**2/k**2])

def logistic_sensitivity_rhs(y_and_dydp, t, r, k):
    y = y_and_dydp[0]
    dydp = y_and_dydp[1:]

    dydt = logistic_rhs(y, t, r, k)
    d_dydp_dt = logistic_dy(y, t, r, k) * dydp + logistic_dp(y, t, r, k)

    return np.concatenate(([dydt], d_dydp_dt))

def plot_logistic(r, k, t):
    p0 = 1e-2
    y = odeint(logistic_rhs, p0, t, args=(r, k))

    y0 = np.zeros(3)
    y0[0] = p0
    sol = odeint(logistic_sensitivity_rhs, y0, t, args=(r, k))
    y = sol[:,0]
    dydr = sol[:,1]
    dydk = sol[:,2]

    y_exact = k / (1 + (k/p0 - 1)*np.exp(-r*t))
    dydr_exact = k*t*(k/p0 - 1) * np.exp(-r*t) / ((k/p0-1)*np.exp(-r*t) + 1)**2
    dydk_exact = -k*np.exp(-r*t) / (p0*((k/p0 - 1)*np.exp(-r*t) + 1.0)**2) + 1.0/((k/p0 - 1)*np.exp(-r*t) + 1)

    plt.clf()
    plt.plot(t, y, label='odeint')
    plt.plot(t, y_exact, label='exact')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(t, dydr, label='odeint')
    plt.plot(t, dydr_exact, label='exact')
    plt.xlabel('x')
    plt.ylabel('dydr')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(t, dydk, label='odeint')
    plt.plot(t, dydk_exact, label='exact')
    plt.xlabel('x')
    plt.ylabel('dydk')
    plt.legend()
    plt.show()

def solve_logistic(r, k, t):
    p0 = 1e-2
    y0 = np.zeros(3)
    y0[0] = p0
    sol = odeint(logistic_sensitivity_rhs, y0, t, args=(r, k))
    return sol


def fit_logistic():
    t = np.linspace(0, 1, 100)
    r = 10.0
    k = 1.0
    p0 = 1e-2
    var = 0.1

    y_data = k / (1 + (k/p0 - 1)*np.exp(-r*t)) + var*np.random.randn(len(t))

    def least_squares(x):
        sol = solve_logistic(x[0], x[1], t)
        y = sol[:, 0]
        dydr = sol[:, 1]
        dydk = sol[:, 2]
        return np.sum((y - y_data)**2)

    def jac(x):
        sol = solve_logistic(x[0], x[1], t)
        y = sol[:, 0]
        dydr = sol[:, 1]
        dydk = sol[:, 2]
        return np.array([
                2*np.sum((y - y_data)*dydr),
                2*np.sum((y - y_data)*dydk),
                ])


    x0 = np.array([1.0, 5.0])
    for method in ['nelder-mead', 'bfgs', 'newton-cg']:
        jac = jac
        res = minimize(least_squares, x0, method=method, jac = jac, options={'disp': True})
        print(res.x)


if __name__ == '__main__':
    fit_logistic()


