---
title: Model Fitting
author: Martin Robinson
date: Nov 2019
urlcolor: blue

\DeclareMathOperator*{\argmin}{arg\,min}
---

# Outline

We will discuss fitting a given dataset $\mathbf{y}$ to two different models:

1. Polynomial model
2. Ordinary Differential Equation model

# Polynomial Regression 

- Your model for the function $y(x)$ is given by a polynomial of degree $n$

  $$ y(x) = \sum_{k=0}^n \beta_k x^k $$

- For a dataset $\mathbf{y}, \mathbf{x}$ of size $m$, this gives a set of $m$ equations:

  $$ y_j = \sum_{k=0}^n \beta_k x^k_j $$

- Which you can combine into a single linear algebra equation:

  $$ \mathbf{y}= M \mathbf{\beta} $$

  Where $M$ is an $m \times n$ matrix with extries $M_{j,k} = x^k_j$

# Solving polynomial regression

- We want to find the vector $\mathbf{\beta}$ such that $\mathbf{\beta} = \argmin_\beta 
  ||y = M \mathbf{\beta}||$, can do this using ordinary least squares

$$ \mathbf{\beta} = (M^TM)^{-1} M^T \mathbf{y}$$

- Ordinary least square can suffer for ill-conditioning for large number of parameters, 
  so we can add small positive elements to the diagonal to improve the reliability of 
  the method (Tikhonov regularization or Ridge Regression):

$$ \mathbf{\beta} = (M^TM + \lambda I)^{-1} M^T \mathbf{y}$$

- Note, a useful alternative to implementing least squares yourself is to use the Scipy 
  function `numpy.linalg.lstsq`, documentation 
  [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html)

# How good is my fitted model?

Can evaluate the goodness of fit of of the model to the data via

1. *Residuals*: that is, $|y - \beta M|$. This gives a poor evaluation of how good the 
   model is at prediction to new data points.

2. *Cross validation*: This is a more direct measure of the predictive power of the 
   model. Divide the model into a *training* set, and a *test* set. Use the training set 
   to calculate $\beta$, and then evaluate the model at the positions of the test set 
   and measure the error

- $k$-fold cross validation divides the dataset into $k$ subsets. For each of $k$ 
  experiments, cross validation is performed using the $k^{th}$ subset as the testing 
  data. The final results are averaged across the $k$ experiments.
- leave-one-out cross validation is the extreme form of $k$-fold with $k=m$


# Fitting ODE models

- Our model is now:

$$\frac{dy}{dx} = f(x, y, \beta)$$

- More difficulte to evaluate than a polynomial model, but can take advantage of 
  domain-specific mechanistic models of the underlying process. The parameters $\beta$ 
  are often physically relevent parameters

- How to fit? Simplest approach is to assume that the data has been sampled from a 
  normal distribution with the mean given by the solution of the ODE:

$$z_j \sim y(x_j, \beta) + \mathcal{N}(0, \sigma)$$

- This give us a likelihood of a given set of parameters $\beta$, given the data 
  $\mathbf{y}$:

$$\mathcal{L}(\beta, \sigma | \mathbf{z}) = \prod_{j}^{m} \frac{1}{\sqrt{2 \pi \sigma^2} 
\exp \left( -\frac{(z_j-y(x_j,\beta))^2}{2 \sigma^2} \right)$$

- Normally work with the log-likelihood:

$$\log \mathbcal{L}(\beta, \sigma | \mathbf{z}) = -\frac{m}{2} \log(2\pi) - m 
\log(\sigma) - \frac{1}{2\sigma^2} \sum_j^m (z_j-y(x_j,\beta))^2$$

- We want to find the parameters $\beta$ that maximise the log-likelihoood. Assuming $m$ 
  and $\sigma$ are fixed, this corresponds to minimising the sum-of-squares of the 
  residuals:

$$\sum_j^m (z_j-y(x_j,\beta))^2$$

# Gradient of the log-likeliood

- We want to minimise the function:

$$f(\beta) = \sum_j^m (z_j-y(x_j,\beta))^2$$

- If we want to use a gradient-based optimisation method, we need the gradient wrt 
  $\beta$

$$\frac{d f(\beta)}{d\beta} = \sum_j^m 
-2\frac{dy}{d\beta}(x_j,\beta)(z_j-y(x_j,\beta))$$



