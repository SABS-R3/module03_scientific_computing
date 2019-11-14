---
title: Finite Difference solution of PDEs
author: Martin Robinson
date: Nov 2019
urlcolor: blue
---


# Finite Differences

Based on Taylor series.  Taylor expand $u(x+h)$ and $u(x-h)$ (and others) in small 
parameter $h$ about $x$. Consider a smooth function $u(x)$ then its Taylor expansion is

$$u(x+h) = u(x) + h u'(x) + \frac{h^2}{2} u''(x) + \frac{h^3}{6} u'''(x) + \frac{h^4}{24} u'''''(x) + \ldots $$

$$u(x-h) = u(x) - h u'(x) + \frac{h^2}{2} u''(x) - \frac{h^3}{6} u'''(x) + \frac{h^4}{24} u'''''(x) - \ldots $$

# Examples

From this can show how we can compute three difference \emph{schemes}
(approximations) to $u'(x)$:

**Forward difference**:
$$u'(x) = \frac{u(x+h)-u(x)}{h} + O(h)$$

**Backward difference**:
$$u'(x) = \frac{u(x)-u(x-h)}{h} + O(h)$$

**Centered difference**:
$$u'(x) = \frac{u(x+h)-u(x-h)}{2h} + O(h^2)$$


# Partial Differential Equations (PDEs)

PDE: partial differential equation: $\ge 2$ independent variables.

History:

1600s: calculus.

1700s, 1800s: PDEs all over physics, especially linear theory.

1900s: the nonlinear explosion and spread into physiology, biology,
     electrical engr., finance, and just about everywhere else.


# The PDE Coffee Table Book

Unfortunately this doesn't physically exist (yet!).  Fortunately, we
can read many of the pages online:
[people.maths.ox.ac.uk/trefethen/pdectb.html](http://people.maths.ox.ac.uk/trefethen/pdectb.html)

# Notation:

* _Partial derivatives_
   $u_t = \frac{\partial u}{\partial t}$,
   $u_{xx} = \frac{\partial^2 u}{\partial x^2},$

* _Gradient_ (a vector)
   $\text{grad}\, u = \nabla u = (u_x, u_y, u_z),$

* _Laplacian_ (a scalar)
   $\text{lap}\, u = \nabla^2 u = \triangle u = \Delta u
	      = u_{xx} + u_{yy} + u_{zz},$

* _Divergence_ (vector input, scalar output)

$\text{div} (u,v,w)^T = \nabla \cdot (u,v,w)^T = u_x + v_y + w_z.$

(and combinations: e.g., $\Delta u = \nabla \cdot \nabla u$).

# Examples

Laplace eq: $\Delta u = 0$                 (elliptic)

Poisson eq: $\Delta u = f(x,y,z)$          (elliptic)

Heat or diffusion eq: $u_t  = \Delta u$    (parabolic)

Wave eq: $u_{tt} = \Delta u$               (hyperbolic)

Burgers eq: $u_t  = (u^2)_x  + \epsilon u_{xx}$

KdV eq: $u_t = (u^2)_x  + u_{xxx}$

# FD approximation of Laplace equation

In 1D, Laplace equation is

$$u_{xx} = 0\text{ for }0 \le x \le 1$$

Central FD approximation of $u_{xx}$ is:

$$u_{xx} \approx \frac{u(x + h) - 2u(x) + u(x-h)}{h^2}$$

# FD discretisation

Discretise $u_{xx} = 0$ in space:

              +----+----+----------+----+> x
              0   x_1  x_2    ... x_N   1

Gives a set of $N$ equations at each interior point on the domain:

$$\frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} = 0 \text{ for } i = 1...N$$

# Boundary conditions 

- Need boundary conditions to solve these equations
- Boundary conditions normally one of two types:
  - **Dirichlet:** A condition on the value of $u$, e.g. $u(0) = 0$
  - **Neumann:** A condition on the derivative of $u$, e.g. $u_x(0) = 0$
- In this case, the boundary conditions act as equations at $x_0$ and $x_{N+1}$

# Dirichlet 

- Take for example $u(0) = 0$ (homogenous dirichlet bc), this gives $u_0 = 0$, and the 
  equation at $x_1$ becomes:

$$\frac{u_{i+1} - 2u_i + 0}{h^2} = 0$$

- Lets put a similar bc at the other boundary, $u(1) = 0$, and represent the final $N$ 
  equations in matrix format:

$$
\begin{bmatrix} -2      & 1      &         &   &     \\
 1      & -2     & 1       &       & \\
&\ddots & \ddots  &  \ddots &\\
&        & 1      &  -2     &  1     \\
&        &        &   1     & -2     \end{bmatrix}
\begin{bmatrix} u_1    \\
u_2    \\
\vdots \\
u_{N-1}\\
u_{N}  
\end{bmatrix}
= \begin{bmatrix} 0    \\
0    \\
\vdots \\
0    \\
0    
\end{bmatrix}
$$

# Non-homogenous Dirichlet bc

- Assume we have the following bcs:

$$u(0) = 1$$
$$u(1) = 0$$

- Can represent this using an additive vector:

$$
\begin{bmatrix} -2      & 1      &         &  &      \\
 1      & -2     & 1       &      &  \\
&\ddots & \ddots  &  \ddots &\\
&        & 1      &  -2     &  1     \\
&        &        &   1     & -2     \end{bmatrix}
\begin{bmatrix} u_1    \\
u_2    \\
\vdots \\
u_{N-1}\\
u_{N}  \end{bmatrix}
+
\begin{bmatrix} 1    \\
0    \\
\vdots \\
0    \\
0    \end{bmatrix}
= \begin{bmatrix} 0    \\
0    \\
\vdots \\
0    \\
0    \end{bmatrix}
$$


# Neumann bc

- Assume we have the following bcs:

$$u_x(0) = 0$$
$$u(1) = 0$$

- bc at $x=0$ gives the following equation (using forward difference):

$$u_1 - u_0 = 0$$

- So the equation at $x_1$ becomes:

$$\frac{u_{i+1} - 2u_i + u_{i}}{h^2} = 0$$

# Neumann bc

- And the equations become:

$$
\begin{bmatrix} -1      & 1      &         &    &   \\
 1      & -2     & 1       &      & \\
&\ddots & \ddots  &  \ddots &\\
&        & 1      &  -2     &  1     \\
&        &        &   1     & -2     \end{bmatrix}
\begin{bmatrix} u_1    \\
u_2    \\
\vdots \\
u_{N-1}\\
u_{N}  \end{bmatrix}
= \begin{bmatrix} 0    \\
0    \\
\vdots \\
0    \\
0    \end{bmatrix}
$$

# Extending to 2D

$$u_{xx} + u_{yy} = 0\text{ for }0 \le x,y \le 1$$

- Discretise both $u_{xx}$ and $u_{yy}$ separatly

$$\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2} + \frac{u_{i,j+1} - 2u_{i,j} + 
u_{i,j-1}}{h^2}= 0 \text{ for } i = 1...N, j=1...N$$

-------------------------------

               ^
          1    +----+----+----------+----+
               |    |    |          |    |
          y_N  +----+----+----------+----+
               |    |    |          |    |
               |    |    |          |    |
               |    |    |          |    |
          y_1  +----+----+----------+----+
               |    |    |          |    |
          y_0  +----+----+----------+----+
               |    |    |          |    |
               +----+----+----------+----+> x
              0   x_1  x_2    ... x_N   1


- To form a matrix equation, collect all the $u_{i,j}$ values into a single vector

----------------------

- Assuming lexiocographical ordering where $i$ advances more quickly than $j$, gives the 
  following equation (with homogenous Dirichlet bcs)

\begin{equation*}
\left[\begin{array}{@{}cccc|cccc@{}}
-4 & 1 &         &        &  1   &      &        &  \\
1  & -4& 1       &        &      & 1    &        &  \\
   & 1 &  -4     &   1    &      &      & \ddots &  \\
   &   &  \ddots & \ddots &\ddots&      &        & \ddots \\
   \hline
  1&   &         & \ddots &\ddots&\ddots&        &  \\
   & 1 &         &        &\ddots&      &        &  \\
   &   & \ddots  &        &      &      &        &   \\
   &   &         & \ddots &      &      &        &   \end{array}\right]
\left[\begin{array}{@{}c@{}}
  u_{1,1}    \\
u_{2,1}    \\
\vdots \\
u_{N,1}  \\
\hline
u_{1,2}    \\
u_{2,2}    \\
\vdots \\
u_{N,2}  \end{array}\right]
= \left[\begin{array}{@{}c@{}}
 0    \\
0    \\
\vdots \\
0    \\
  \hline  
  0    \\
0    \\
\vdots \\
0   \end{array}\right]
\end{equation*}

# Kronicker product

- This matrix $A_{2D}$ is acutally the sum of two terms, one associated with the 
  discretisation of $u_{xx}$ given by $A_x$, and the other associated with the 
  discretisation of $u_{yy}$ given by $A_y$
- We can use the Kronecker (tensor) product to construct $A_{2D}$

$$A_{2D} = (I_y \otimes A_x) + (A_y \otimes I_x)$$

- Where the Kronecker product $C = A \otimes B$ is given by


# Solving linear systems

- Each set of equations are a set of linear equations of the form:

$$A u = b$$

- Can find the solution $u$ via any direct linear solver. Most basic method is Gaussian 
  Elimination, many others exist based on different factorisations of the matrix $A$
    - lower-upper (LU) decomposition
    - QR decomposition
    - Cholesky decomposition


# Solving linear systems with Scipy

- Scipy has a linear algebra solver for *dense* matrices: `scipy.linalg.solve` 
  [(docs)](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.solve.html)
    - Uses LU factorization
- Scipy also has a linear algebra solver for *sparse* matrices: `scipy.sparse.spsolve` 
  [(docs)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve)
    - I believe this uses a sparse LU factorisation provided by SuperLU
- Also other specialised methods to solve [triangular 
  systems](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html), 
  or to directly compute [matrix 
  factorisations](https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/linalg.html)
- Later on in the course you will cover iterative solvers as well.

# Sparse versus Dense matrices

- The matrix $A$ from a finite difference discretisation is often very *sparse* (i.e. 
  contains many zeros)
- Scipy has many different sparse matrix formats that only store non-zero elements to 
  save both memory and computation. For more details see the `scipy.sparse` 
  [module](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- Often using sparse instead of dense matrices for large systems can save you 
  significant computational effort, scalings are $\mathcal{O}(N)$ instead of 
  $\mathcal{O}(N^2)$

# The end - to the exercises!
