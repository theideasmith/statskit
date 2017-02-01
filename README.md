# statskit
A micro statistics library I'm working on as I take my intro stats class

## Usage

Let's get everything setup
```python
import Statskit as stats
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
```
First, we'll sample from a normal distribution
```python
x = random.normal(2, 0.4, 500)
```

## PMF
Statskit computes binned frequencies
```python
F, bins= stats.make_histogram(x, 100) 
```
Which you can normalize
```python
P = stats.normalized(F)    
```
Package everything nicely with the `stats.DiscreteRandomVariable` data structure

```
X = stats.DiscreteRandomVariable(x, P, bins)     
```

Calculate the probability `X` falls in a range using the `P` operator

```python
X.P[0.2:3.5]
```

## Expectation

You can compute the expectation of a random variable with respect to a function. Due to the way statskit works, any function you pass must be given as its antiderivative. So if you wanted to compute `E[f]` you must use `F` as the function passed. 

```
expectation = X.E[ F ]
```

Statskit has already implemented expected: 

- `emean()`
- `evariance()`

Which you can use as:

```python
X.E[ emean() ]
```
```python
X.E[ evariance() ]
```

## Plots

Now, let's do a plot
```python
# Data to plot the cdf 
rs = np.arange(0, 4.5, 0.05)       
# Generating the pdf
vs = np.array([stats.cumulative(0,D, i) for i in rs])   

fig, ax = plt.subplots(2, sharex=True)
fig.suptitle("Some awesome plots", fontsize=20)

ax[0].set_title('CMF', fontsize=15)
ax[0].scatter(rs, vs)
ax[0].set_xlim(0, 4.)

ax[1].set_title("PMF", fontsize=15)
ax[1].bar(bins[:-1], F.tolist(), width=bins[1:] - bins[:-1])
ax[1].set_xlim(0, 4.)
```
 
 ![Image](https://raw.githubusercontent.com/theideasmith/statskit/master/dist.jpg)
