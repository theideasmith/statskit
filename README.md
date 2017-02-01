# statskit
A micro statistics library I'm working on as I take my intro stats class

## Usage

```python
import Statskit as stats
from numpy import random
import matplotlib.pyplot as plt
import numpy as np

# Sampling from a distribution
X = random.normal(2, 0.4, 500)

# Generating frequency as pmf
F, bins= stats.make_histogram(X, 100)               
P = stats.normalized(F)    

# Packaging the data together into the DiscreteRandomVariable class
D = stats.DiscreteRandomVariable(X, P, bins)     

# Data to plot the cdf 

rs = np.arange(0, 4.5, 0.05)                         
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
 
 ![Image](https://raw.githubusercontent.com/theideasmith/statskit/master/dist.png)
