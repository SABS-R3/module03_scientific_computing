---
title: Time integration of PDEs
author: Martin Robinson
date: Nov 2019
urlcolor: blue
---

# Finite differences in space and time

The simplest approach to numerical soln of PDE is finite difference
discretization in both space and time (if it's time-dependent).

Consider the heat or diffusion equation in 1D:


$$u_t = u_{xx},  \,\,\,    -1 < x < 1,$$

with _initial conditions_:  $u(x,0) = u_0(x)$,

and _boundary conditions_:  $u(-1,t) = u(1,t) = 0$.

# Finite differences in space and time

Set up a regular grid with  $k$ = time step ,  $h$ = spatial step

$$v^n_j  \approx  u(x,t).$$

A simple finite difference formula:

$$\frac{v^{n+1}_j  -  v^n_j}{k}  =  \frac{v^n_{j+1} - 2v^n_j + v^n_{j-1}}{h^2}.$$

# Linear Algebra

- Can rearrange the FD formula in the previous slide to a linear algebra expression

$$
\begin{bmatrix} v^{n+1}_1    \\
v^{n+1}_2    \\
\vdots \\
v^{n+1}_{N-1}\\
v^{n+1}_{N}  \end{bmatrix}
=
\begin{bmatrix} v^{n}_1    \\
v^{n}_2    \\
\vdots \\
v^{n}_{N-1}\\
v^{n}_{N}  
\end{bmatrix}
+ \frac{k}{h^2}
\begin{bmatrix} -2      & 1      &         &   &     \\
 1      & -2     & 1       &       & \\
&\ddots & \ddots  &  \ddots &\\
&        & 1      &  -2     &  1     \\
&        &        &   1     & -2     \end{bmatrix}
\begin{bmatrix} v_1    \\
v^{n}_2    \\
\vdots \\
v^{n}_{N-1}\\
v^{n}_{N}  \end{bmatrix}
$$

- or

$$\mathbf{v}^{n+1} = \mathbf{v}^{n} + k L \mathbf{v}^{n}$$
$$\mathbf{v}^{n+1} = (I + k L) \mathbf{v}^{n}$$


# The Method-of-lines

Discretize in space only, to get a system of ODEs.  This reduces the
problem to one we've already looked at: numerical solution of ODEs.

For the heat/diffusion equation, we get:

  $$\frac{d}{dt} v_j = \frac{ v_{j-1} - 2v_j + v_{j+1} }{h^2},$$

where $v_j(t)$ is a continuous function in time and $v_j(t) \approx
u(x_j,t)$.


---------------------

We solve the (coupled) ODEs along lines in time.

             t
             ^
             |
             |  ^    ^    ^          ^
             |  |    |    |          |
             |  |    |    |          |
             |  |    |    |          |
             +--+----+----+----------+----> x
               x_0  x_1  x_2    ... x_N



# Linear algebra: MOL approach

$$
\frac{d}{dt}
\begin{bmatrix} v_1(t)    \\
v_2(t)    \\
\vdots \\
v_{N-1}(t)\\
v_{N}(t)  \end{bmatrix}
=
\frac{1}{h^2}
\begin{bmatrix} -2      & 1      &         &   &     \\
 1      & -2     & 1       &       & \\
&\ddots & \ddots  &  \ddots &\\
&        & 1      &  -2     &  1     \\
&        &        &   1     & -2     \end{bmatrix}
\begin{bmatrix} v_1(t)    \\
v_2(t)    \\
\vdots \\
v_{N-1}(t)\\
v_{N}(t)  \end{bmatrix}
$$


- Or using $v(t)$ as $N$-vector:

$$\frac{d}{dt} \mathbf{v}(t) = L \mathbf{v}(t)$$

------------------------------

- If we then discretize time with forward Euler we get:

$$\mathbf{v}^{n+1} = \mathbf{v}^{n} + k L \mathbf{v}^n$$
$$\mathbf{v}^{n+1} = (I + k L) \mathbf{v}^n$$


where $v^{n+1}$ represents the discrete vector of solution at time $t_{n+1}$:

$$v^{n+1} \approx v(t_{n+1}).$$

- This is the same as the earlier fully discrete equations 
  - Note however that not all finite difference schemes are methods-of-lines.

# Stability in finite difference calculations

- A fourth-order problem (biharmonic equation)

$$u_t = -u_{xxxx}.$$

How to discretize?  Think $(u_{xx})_{xx}$..., this leads, eventually, to

  $$v^{n+1}_j = v^n_j - \frac{k}{h^4} \left(
    v^n_{j-2}  - 4v^n_{j-1}  + 6v^n_j  - 4v^n_{j+1}  + v^n_{j+2}
  \right)$$
or
  $$\frac{d}{dt} v_j = \frac{1}{h^4} \left(
    v_{j-2}  - 4v_{j-1}  + 6v_j  - 4v_{j+1}  + v_{j+2}
  \right)$$
  $$\frac{d}{dt} \mathbf{v} = -H\mathbf{v}$$

- Note: if you already have the FD approximation of $L \approx u_{xx}$, can construct 
  this as $H = LL$

- Using explicit Euler, this leads to rediculously small timesteps, otherwise the 
  problem becomes *unstable*

# von Neumann stability analysis

One approach is *von Neumann Analysis* of the finite difference
formula, also known as *discrete Fourier analysis*, invented in the
late 1940s.

Suppose we have periodic boundary conditions and that at step $n$ we
have a (complex) sine wave

  $$v_j^n  = \exp(i \xi x_j ) = \exp(i \xi j h),$$

for some wave number $\xi$.  Higher $\xi$ is more oscillatory.  

We will analysis whether this wave grows in amplitude or decays (for each
$\xi$).  For stability, we want all waves to decay.


---------------------

For the biharmonic diffusion equation, we substitute this wave into
the finite difference scheme above, and factor out $\exp(i \xi h)$ to get

  $$v^{n+1}_j = g(\xi) v^n_j,$$

with the *amplification factor*

  $$ g(\xi) = 1 - \frac{16k}{h^4} \sin (\xi \, h/2). $$

A mode will blow up if  $|g(\xi)| > 1$.  Thus for stability we want
to ensure  $|g(\xi)| \leq 1$, i.e.,

  $$ 1 - 16k / h^4 \geq -1,
  \qquad\text{or \qquad $\boxed{  k \leq h^4 / 8.  }$}$$

For $h = 0.025$, this gives
  $\boxed{  k  \leq  4.883e-08  }$, and confirms that this finite difference formula is 
  not really practical.


# Implicit methods for PDEs

Apply implicit (A stable) methods to the semidiscrete method-of-lines
form.  For example, let's look at the heat equation $u_t = \Delta u$.
If we apply backward Euler:

  $$\frac{\mathbf{v}^{n+1} - \mathbf{v}^n}{k} = L \mathbf{v}^{n+1},$$
  $$(I - k L) \mathbf{v}^{n+1}  = \mathbf{v}^n$$
  $$A \mathbf{v}^{n+1}  = \mathbf{v}^n$$

- Solve linear system to find $\mathbf{v}^{n+1}$
- Backward Euler is unconditionally stable


# IMEX discretization

- IMEX (implicit/explicit) discretizations can be used for stiff equations that contain 
  non-linear terms, e.g. 

$$u_t(t) = u_{xxxx}(t) + u^2$$

- We treat the linear stiff terms implicitly and the nonlinear (but
hopefully nonstiff) terms explicitly

$$\frac{v^{n+1} - v^n}{k} = -H v^{n+1} + v^n \circ v^n$$


# The end - to the exercises!
