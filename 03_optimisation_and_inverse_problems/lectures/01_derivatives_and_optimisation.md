---
title: Derivatives and Optimisation in Python
author: Martin Robinson
date: Nov 2019
urlcolor: blue
---

# Outline

1. Calculating Derivatives
  - Symbolic
  - Numerical
  - Automatic

2. "Black-box" Optimisation
  - Scipy optimize module

# Differentiation in Optimisation

- Assume the function $f(x)$ you wish to optimise is differentiable, then we can 
  consider the equation of a tangent line at a given point $x_n$

  $$ y(x) = f(x_n) + f'(x_n) (x - x_n)$$

- We want to find the value of $x_{n+1}$ such that $y(x_{n+1}) = 0$, i.e.

  $$ 0 = f(x_n) + f'(x_n) (x_{n+1} - x_n)$$

- Rearange to get:

  $$ x_{n+1} = x+n - \frac{f(x_n)}{f'(x_n)}$$

--------------------------

- This is Newton's method for root finding, repeatedly find $x_{n+1}$ until converged to 
  the root of the original function $f(x) = 0$. In what case will you reach the true 
  root after a single iteration?
- EAsy to extend to minimisation/maximisation of $f(x)$. We know that local 
  minima/maxima exist when $f'(x) = 0$, so can apply Newton's method to the derivative 
  instead:

  $$ x_{n+1} = x+n - \frac{f'(x_n)}{f''(x_n)}$$

- Large, important class of optimisation methods use gradient information to efficiently 
  find the minima/maxima of a differentiable function:
    - Gradient Descent, Conjugate gradient descent, stochastic gradient descent etc.
    - Root finding methods like Newton's, Levenberg-Marquardt, Quasi-Newton methods like 
      Broyden-Fletcher-Goldfarb-Shanno (BFGS) etc.

# Symbolic Differentiation


$$ f'(x) = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h} $$

- Can calculate the derivative of a given function $f$ by hand, or use a symbolic 
  package. Latter is often less error-prone...

- [Sympy](https://www.sympy.org/en/index.html) in Python, but more powerful specialised 
  packages (e.g. [Mathematica](https://www.wolfram.com/mathematica/)) exist.

# Sympy

- A symbolic package like Sympy allows you to construct a tree-based representation of 
  an equation

```python
import sympy as sp

x = sp.Symbol('x')

f = 2*x + 1
```

![equation f given by](figs/sympy_test.pdf)

# Sympy diff

- The `diff()` function in Sympy calculates the derivative of an expression tree using 
  standard derivative rules (e.g. chain rule, product rule etc.)
- Results in another expression tree

```python
f = 2*x + 1
df = f.diff()
```

![equation f given by](figs/sympy_test_diff.pdf)

# Numerical Differentiation

Finite Difference approximation to derivatives can be constructed from the Taylor series
expansion:

$$
f(x_0 + h) = f(x_0) + h f'(x_0) + \frac{h^2}{2!} f''(x_0) + \frac{h^3}{3!} f'''(x_0)
$$

leads to:

$$\text{Forward Difference: }f'(x_0) = \frac{f(x_0 + h) - f(x_0)}{h} + \mathcal{O}(h)$$
$$\text{Backwards Difference: }f'(x_0) = \frac{f(x_0 + h) - f(x_0)}{h} + 
\mathcal{O}(h)$$
$$\text{Central Difference: }f'(x_0) = \frac{f(x_0 + h) - f(x_0 - h)}{2h} + 
\mathcal{O}(h^2)$$

# Automatic Differentiation

- Automatic differentiation is the application of the Chain Rule to a computer program
- Imagine a Python function with an input $x$ and result $y$. Statements in this 
  function might consist of:

\begin{align*}
  w_0 &= x \\
  w_1 &= h(w_0) \\
  w_2 &= g(w_1) \\
  w_3 &= f(w_2) \\
  y &= w_3 \\
\end{align*}

- This is the same as:

$$
y = f(g(h(x)))
$$

- And the chain rule gives 


$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial w_3} \frac{\partial 
w_3}{\partial w_2} \frac{\partial w_2}{\partial w_1} \frac{\partial w_1}{\partial w_0} 
\frac{\partial w_0}{\partial x}
$$


# Forward mode AD

- Evaluate the derivative from right to left:

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial w_3} \left( \frac{\partial 
w_3}{\partial w_2} \left( \frac{\partial w_2}{\partial w_1} \left( \frac{\partial 
w_1}{\partial w_0} \left( \frac{\partial w_0}{\partial x} \right) \right) \right) 
\right)
$$

- This corresponds to the natural flow of the program (i.e. top to bottom)
- To compute, each program statement is augmented to also calculate its derivative with 
  $x$
- Require one pass of forward mode AD for each input, efficient for functions with more 
  inputs than outputs.

# Forward mode AD example 

Note: taken from [wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)

$$y = f(x_0, x_1) = x_0 x_1 + \sin(x_1)$$

\begin{center}
\begin{tabular}{ |c|c| } 
  
  Evaluate $y$ & Evaluate $\partial y / \partial x_0$ \\

  $w_0 = x_0$ & $w_0' = 1$ \\ 

  $w_1 = x_1$ & $w_1' = 0$ \\

  $w_2 = w_0 w_1$ & $w_2' = w_0' w_1 + w_0 w_1'$ \\

  $w_3 = \sin w_0$ & $w_3' = w_0' \cos w_0$ \\

  $w_4 = w_2 + w_3$ & $w_4' = w_2' + w_3'$\\

  $y = w_4$ & $y' = w_4'$

\end{tabular}
\end{center}


# Reverse mode AD

- Evaluate the derivative from left to right:

$$
\frac{\partial y}{\partial x} = \left( \left( \left( \left( \frac{\partial y}{\partial 
w_3} \right) \frac{\partial w_3}{\partial w_2} \right) \frac{\partial w_2}{\partial w_1} 
\right) \frac{\partial w_1}{\partial w_0} \right) \frac{\partial w_0}{\partial x}  $$

- Now the quantity of interest is the adjoint $\overline{w} = \frac{\partial y}{\partial 
  w}$
- Corresponds to the reverse flow of the program. Need to keep a record of what 
  operations have been performed, and in which order (i.e. reverse mode AD execution 
  occurs from bottom to top)
- Computational cost scales with the number of *outputs*, efficient for functions with 
  more inputs than outputs.

# Reverse mode AD example 

Note: taken from [wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)

$$y = f(x_0, x_1) = x_0 x_1 + \sin(x_1)$$

1. Evaluate $y$:
  \begin{align*}
    w_0 &= x_0  \\
    w_1 &= x_1 \\
    w_2 &= w_0 w_1\\
    w_3 &= \sin w_0\\
    w_4 &= w_2 + w_3
  \end{align*}

---------------------

2. Evaluate $\partial y / \partial x$:
 \begin{align*}
   \overline{w}_4 &= 1 \\
   \overline{w}_3 &= \overline{w}_4 \\
   \overline{w}_2 &= \overline{w}_4 \\
   \overline{w}_1 &= \overline{w}_2 w_0 \\
   \overline{w}_0 &= \overline{w}_3 \cos w_0 + \overline{w}_2 w_1
  \end{align*}


# Scipy Optimisation

- Scipy has a useful module 
  [`optimise`](https://docs.scipy.org/doc/scipy/reference/optimize.html), that includes 
  a function `minimize` that performs local minimisation of scalar functions

```python
from scipy.optimize import minimize

def quadratic(x):
  return x**2
  
res = minimize(quadratic, 1, method='BFGS')
print('minima is at {}'.format(res.x))
```

# Local Minimisation

Given a starting point $\mathbf{x}_0$, find the nearest local minima of some function 
$f(\mathbf{x})$

- *Simplex methods*: based on evolving a simplex, or geometrical volume, through 
  subsequent iterations. No gradients required. E.g. Nelder-Mead
- *Gradient-based methods*: gradient descent, conjugate gradients, etc.
- *Root-finding methods*: Newton's,  Broyden-Fletcher-Goldfarb-Shanno (BFGS)

# Global minimisation:

Given a starting point $\mathbf{x}_0$, find the global minima of some function 
$f(\mathbf{x})$

- A difficult problem, need to avoid local minimia.
- Simplest method is to restart a local method at many points in space.
- Many, many (order of 100's) proposed methods, Scipy implements a few (basinhopping, 
  differential_evolution, shgo, dual_annealing).
- Computational Biology group in Computer Science maintain the 
  [PINTS](https://github.com/pints-team/pints) python package, which contain a few more 
  (CMA-ES, an evolutionary algorithm similar to differential evolution, is particularly 
  reliable)


