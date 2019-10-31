import numpy as np
import scipy
import matplotlib.pylab as plt
import pandas as pd

x = np.arange(1, 20).reshape(-1, 1)
intercept   = -10
slope       = 2.5
y = intercept + slope*x
y = y + 10*np.random.randn(len(y),1) # add some noise

plt.figure()
plt.plot(x,y,'o')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


M1 = x                                           # w/o intercept
M2 = np.concatenate([np.ones((len(x),1)), x], axis=1) # w/ intercept

beta1 = np.linalg.pinv(M1) @ y
beta2 = np.linalg.pinv(M2) @ y

plt.clf()
plt.grid(True)
plt.plot(x,y,'o')
plt.plot(x, M1 @ beta1, 'r-')
plt.plot(x, M2 @ beta2, 'b-')
plt.show()

M3 = np.concatenate([np.ones((len(x),1)), x-np.mean(x)], axis=1)
beta3 = np.linalg.pinv(M3) @ y
print(beta2)
print(beta3)


names = ['year', 'month', 'maxTemp', 'minTemp', 'hoursFrost', 'rain', 'hoursSun']
df = pd.read_csv('OxfordWeather.txt', delim_whitespace=True, header=None, names=names)

meanTemp   = .5*(df.minTemp+df.maxTemp)

plt.close()
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
decades = (np.round(df.year/10)*10)
ax1.boxplot([df.rain[decades == i] for i in decades.unique()],
        positions=decades.unique(),
        widths=5)
ax1.set_xlabel('decade')
ax1.set_ylabel('rain')

ax2.boxplot([df.rain[df.month==i] for i in df.month.unique()],
        positions=df.month.unique())
ax2.set_xlabel('month')
ax2.set_ylabel('rain')

ax3.plot(df.minTemp, df.maxTemp, '.k')
ax3.set_xlabel('minTemp')
ax3.set_ylabel('maxTemp')

ax4.plot(meanTemp, df.hoursFrost,'.k')
ax4.set_xlabel('meanTemp')
ax4.set_ylabel('hoursFrost')

temps = np.round(meanTemp/3)*3
ax5.boxplot([df.rain[temps==i] for i in temps.unique()],
    positions=temps.unique())
ax5.set_xlabel('meanTemp')
ax5.set_ylabel('rain')

ax6.boxplot([df.hoursSun[df.month==i] for i in df.month.unique()],
    positions=df.month.unique())
ax6.set_xlabel('month')
ax6.set_ylabel('hoursSun')

plt.show()

# Fit linear model ( y=a*x )
x = df.minTemp.values.reshape(-1,1)
y = df.maxTemp.values.reshape(-1,1)

M = x
beta = np.linalg.pinv(M) @ y

plt.clf()
plt.plot(x, y, 'o')
plt.plot(x, M @ beta,'r')
plt.xlabel('minTemp')
plt.ylabel('maxTemp')
plt.show()

# Need a constant? ( y=a*x+b )   [what is b?]
M = np.concatenate([np.ones((len(x),1)), x], axis=1)
beta = np.linalg.pinv(M) @ y
plt.plot(x, M @ beta, 'g')

# Fit quadratic model ( y=a*x^2+b*x+c )
x = df.month.values.reshape(-1,1)
y = df.hoursSun.values.reshape(-1,1)

i = np.argsort(x,axis=0).reshape(-1)  # we have to sort the data according to x if we want a nice plot
x = x[i]
y = y[i]

M = np.concatenate([np.ones_like(x), x, x**2], axis=1)  # this is the cunning bit
beta = np.linalg.pinv(M) @ y
plt.clf()
plt.plot(x, y, 'o')
plt.plot(x, M @ beta, 'r')
plt.xlabel('month')
plt.ylabel('hoursSun')
plt.show()

# Fit exponential model ( y=a*exp(b*x) )

x = df.minTemp.values.reshape(-1,1)
y = np.log(df.hoursFrost.values.reshape(-1,1)) # trick: in order to fit exp with GLM, use log transform!

# unfortunately, this trick only works for hoursFrost&gt;0. So we need to
# exclude data points where hoursFrost=0:
x = x[df.hoursFrost > 0]
y = y[df.hoursFrost > 0]

i = np.argsort(x, axis=0).reshape(-1) # again, easier to plot if we sort x and y
x = x[i]
y = y[i]

M = np.concatenate([np.ones_like(x), x], axis=1)
beta = np.linalg.pinv(M) @ y

plt.clf()
plt.plot(meanTemp, df.hoursFrost, 'o')
plt.plot(x, np.exp(M @ beta), 'r')
plt.xlabel('temp')
plt.ylabel('hoursFrost')
plt.show()

# Note: the exponential model doesn't quite work for this
# data. Below we will try a polynomial model

def fit_poly(x,y,p):
    """
    [beta,pred,err,sse]=fit_poly(x,y,p)

    p is the degree of the polynomial
    """

    M = np.ones_like(x)
    for i in range(1,p):
        xx = x**i
        xx = xx - np.mean(xx)
        xx = xx / np.linalg.norm(xx)
        M  = np.concatenate([M, xx], axis=1)

    beta = np.linalg.pinv(M) @ y   # regression coefficient

    pred = M @ beta       # data predicted by the model
    err  = y-M @ beta     # error
    sse  = np.sum(err**2) # sum of squared error
    return [beta,pred,err,sse]

x = meanTemp.values.reshape(-1,1)
y = df.hoursFrost.values.reshape(-1,1)

i = np.argsort(x, axis=0).reshape(-1)
x = x[i]
y = y[i]

plt.clf()
plt.plot(x, y, 'o')
plt.xlabel('meanTemp')
plt.ylabel('hoursFrost')

[beta, pred, err, sse] = fit_poly(x, y, 5)
plt.plot(x, pred, 'r', lw=2)
plt.show()

#y = a * np.exp(b*x + c*x**2)  # unknown:a,b,c : log transform the data to get GLM
#y = a * np.cos(x + phi)       # unknown a,phi : use trigonometric equalities to get GLM
#y = 1 / (1 + a*np.exp(b*x))   # unknown a,b   : log-transform

x   = np.linspace(0,10,40).reshape(-1,1)
a   = 5     # amplitude parameter
phi = np.pi/4 # phase parameter

y   = a * np.cos(x+phi)            # generate data using cosine model
y   = y + np.random.randn(len(y))  # add noise

plt.clf()
plt.plot(x, y, 'o')

# the trick:
# a*cos(x+phi) = a*np.cos(x)*np.cos(phi) - a*np.sin(x)*np.sin(phi)
#              = [a*cos(phi)]*(cos(x)) + [a*sin(phi)]*(-sin(x))
# let A=a*cos(phi) and B=a*sin(phi)
# which can be inverted thusly: a=norm([A;B]) and phi=atan2(B/a,A/a)
#
# We now have a linear model for A and B: y = A*cos(x)+B*(-sin(x))


M    = np.cos(x) - np.sin(x)
beta = np.linalg.pinv(M) @ y

a_hat   = np.linalg.norm(beta)
phi_hat = np.arctan2(beta[1]/a_hat,beta[0]/a_hat)

plt.plot(x, a_hat*np.cos(x+phi_hat), 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

