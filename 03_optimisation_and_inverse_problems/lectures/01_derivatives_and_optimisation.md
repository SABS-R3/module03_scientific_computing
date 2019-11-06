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

- Familiar from high school math:

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

