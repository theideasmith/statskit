# coding: utf-8

# In[100]:

import Statskit as stats
reload(stats)
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from numpy import *
import numpy as np


numbin = 10
X = random.normal(2, 0.4, 500)                
F, bins= stats.make_histogram(X, numbin)               
P = normalized(F)                             
D = stats.DiscreteRandomVariable(X, P, bins)        
rs = arange(0.05, 4.5, 0.05)                  
vs = array([stats.cumulative(0,D, i) for i in rs])  


fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True,
                               figsize=(7, 7))

fig.suptitle('Statskit Test Plots', fontsize=20)

ax[1].set_title("Numpy native histogram")
ax[1].hist(X, bins=numbin)
ax[1].set_xlim(0, 4.)

xticks = ax[1].get_xticks()
labels = ax[1].get_xticklabels()

ax[0].set_title("Statskit cdf")
ax[0].scatter(rs, vs)
ax[0].set_xlim(0, 4.)

ax[2].set_title("Statskit Hist")
ax[2].bar(bins[:-1], F.tolist(), width=bins[1:] - bins[:-1])
ax[2].set_xlim(0, 4.)

#Configuration
fig.subplots_adjust(hspace=.5)
for a in ax: 
    a.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        right='off',
        left='off',
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)

    

plt.show()


get_ipython().magic(u'pinfo digitize')


