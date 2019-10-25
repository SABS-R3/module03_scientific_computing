import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import graphviz

a = 1.0
b = 100.0


def rosenbrock(x, y):
    return (a-x)**2 + b*(y-x**2)**2

def rosenbrock_diff_x(x, y, h):
    return (rosenbrock(x+h, y) - rosenbrock(x-h, y))/(2*h)

def rosenbrock_diff_y(x, y, h):
    return (rosenbrock(x, y+h) - rosenbrock(x, y-h))/(2*h)

def rosenbrock_diff_FMD(w1, w1_dot, w2, w2_dot):
    w3 = a - w1
    w3_dot = -w1_dot
    w4 = w1**2
    w4_dot = 2*w1*w1_dot
    w5 = w3**2
    w5_dot = 2*w3*w3_dot
    w6 = w2 - w4
    w6_dot = w2_dot - w4_dot
    w7 = w6**2
    w7_dot = 2*w6*w6_dot
    w8 = b*w7
    w8_dot = b*w7_dot
    w9 = w5 + w8
    w9_dot = w5_dot + w8_dot
    return w9_dot

def rosenbrock_diff_x_FMD(x, y):
    w1 = x
    w1_dot = 1.0
    w2 = y
    w2_dot = 0.0
    return rosenbrock_diff_FMD(w1, w1_dot, w2, w2_dot)

def rosenbrock_diff_y_FMD(x, y):
    w1 = x
    w1_dot = 0.0
    w2 = y
    w2_dot = 1.0
    return rosenbrock_diff_FMD(w1, w1_dot, w2, w2_dot)

def rosenbrock_diff_RMD(x, y):
    w1 = x
    w2 = y
    w3 = a - w1
    w4 = w1**2
    w5 = w3**2
    w6 = w2 - w4
    w7 = w6**2
    w8 = b*w7
    w9 = w5 + w8

    w9_bar = 1.0
    w8_bar = w9_bar * 1.0
    w7_bar = w8_bar * b
    w6_bar = w7_bar * 2*w6
    w5_bar = w9_bar * 1.0
    w4_bar = w6_bar * (-1.0)
    w3_bar = w5_bar * 2*w3
    w2_bar = w6_bar * 1.0
    w1_bar = w3_bar * (-1) + w4_bar * 2*w1

    return w1_bar, w2_bar


def symbolic_numerical_compare():
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    eq = rosenbrock(x, y)

    eval_x = 0.5
    eval_y = 0.5

    symbolic_diff_x = eq.diff(x).subs([(x, eval_x), (y, eval_y)])
    symbolic_diff_y = eq.diff(y).subs([(x, eval_x), (y, eval_y)])

    array_of_h = np.linspace(1e-5, 1e-2, 100)

    error_diff_x = [
        abs(symbolic_diff_x - rosenbrock_diff_x(eval_x, eval_y, h))
            for h in array_of_h
    ]

    error_diff_y = [
        abs(symbolic_diff_y - rosenbrock_diff_y(eval_x, eval_y, h))
            for h in array_of_h
    ]

    plt.loglog(array_of_h, error_diff_x, label='diff x')
    plt.loglog(array_of_h, error_diff_y, label='diff y')
    plt.loglog(array_of_h, array_of_h**2, label='$h^2$')
    plt.legend()
    plt.show()

def symbolic_automatic_compare():
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    eq = rosenbrock(x, y)

    dot_str = sp.printing.dot.dotprint(eq)
    dot = graphviz.Source(dot_str)
    dot.render('test.gv',view=True)

    eval_x = 0.5
    eval_y = 0.5

    symbolic_diff_x = eq.diff(x).subs([(x, eval_x), (y, eval_y)])
    symbolic_diff_y = eq.diff(y).subs([(x, eval_x), (y, eval_y)])

    fm_auto_diff_x = rosenbrock_diff_x_FMD(eval_x, eval_y)
    fm_auto_diff_y = rosenbrock_diff_y_FMD(eval_x, eval_y)

    rm_auto_diff_x, rm_auto_diff_y = rosenbrock_diff_RMD(eval_x, eval_y)

    print('error in rosenbrock FM derivative wrt x is {}'
            .format(fm_auto_diff_x - symbolic_diff_x))

    print('error in rosenbrock FM derivative wrt y is {}'
            .format(fm_auto_diff_y - symbolic_diff_y))

    print('error in rosenbrock RM derivative wrt x is {}'
            .format(rm_auto_diff_x - symbolic_diff_x))

    print('error in rosenbrock RM derivative wrt y is {}'
            .format(rm_auto_diff_y - symbolic_diff_y))



if __name__ == '__main__':
    symbolic_automatic_compare()
