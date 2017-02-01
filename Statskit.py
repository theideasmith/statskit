from numpy import *
import matplotlib.pyplot as plt
import numpy as np
class DiscreteRandomVariable:
  def __init__(self,X, P, bins):
    self.X = X
    self.P = P
    self.H = P*X.shape[0]
    self.bins = bins
    
def make_histogram(x, bins): 
  """
  Histograms by range dx
  """
  x1 = min(x)
  x2 = max(x)
  dx = (x2-x1)/bins
  starts = array([arange(x1, x2, dx)]).T
  ends = starts + dx
  m = starts.shape[0]
  
  # This gets us the in each interval starts < a < ends
  xprime = tile(x, (m, 1))

  a = xprime >= broadcast_to(starts, xprime.shape)
  b = xprime < broadcast_to(ends, xprime.shape)

  frequencies = (a*b).sum(1)
  retbins = ones(len(starts)+1)
  retbins[:len(starts)]*=starts.T[0]
  retbins[-1] = ends[-1]
  return frequencies, retbins 

def normalized(h):
  """
  Discrete normalized frequency dataset
  on a histogram h
  """
  px = h/float(sum(h))   
  return px 

def expectation(F, X):
  """
  Normally, 
  E[ f ] = sum{0->inf}{p_ix_i}

  However, because we are dealing 
  with a discrete distribution, 
  useful to have a probability for each 
  individual value. It is much more useful to 
  have binned probabilities. 
  
  Discrete expected value on a dataset 
  Let the nth bin be denoted by        
    B_i = (a_i, b_i)                   
  The density:
    lambda_i = l_i = P( x in B_i)/(b_i - a_i)
  Therefore, we use the following equation to 
  calculate the expected value of a function:

  E[ f ] = sum{0->n}{ l_i*int{ B_ia->B_ib }{ f(x) dx }}
         = sum{0->n}{ l_i*( F(B_ib)-F(B_ia) )} 

  The actual code is simpler than its specification
  """
  
# The probability density function lambda=l
  l = X.P/(X.bins[1:] - X.bins[:-1])
  return sum( l *(F(X.bins[1:]) - F(X.bins[:-1])) )

def mean(D):
  """
  Mean defined in terms of expected value
  """                                                        
  return expectation( lambda y: (1./2)*pow(y,2), D)
  
def variance(D):
  """
  Variance defined in terms of expected value
  """
  u = mean(D)
  return expectation( lambda y: (1./3)*pow(y-u,3), D)
 
def stddev(D):
  return sqrt( variance(D) )

def cumulative(a,X,b):
  bins = array(X.bins)
  if a ==b: 
    return 0
  if b<a:  
    print "INPUTERRROR"
    return 

# Get the intervals
  argbinmin = np.digitize(a, bins) 
  argbinmax = np.digitize(b, bins) 
  if (argbinmin == 0 and argbinmax ==0) or (argbinmax == len(bins) and argbinmin == len(bins)):
    return 0

  if argbinmax == 0:
    argbinmax = 1
  if argbinmin == 0:
    argbinmin = 1

  if argbinmax == len(bins):
    argbinmax = len(bins) -1
  if argbinmin == len(bins):
    argbinmin = len(bins) -1 

  # Here we do a little bit of nasty interpolation
  # that could be vectorized if I had more brain density
  inda = argbinmin-1  
  indb = argbinmax     
   
  Ps = array(X.P[inda:indb] )
  pmin = X.P[argbinmin-1]
  pmin_percentage =(bins[argbinmin]-a)/(bins[argbinmin]-bins[argbinmin-1])

  pmax = X.P[argbinmax-1]
  pmax_percentage = (b-bins[argbinmax-1])/(bins[argbinmax]-bins[argbinmax-1])
  
  Ps[0] = pmin*pmin_percentage
  Ps[-1] = pmax*pmax_percentage

  return sum( Ps )

if __name__=="__main__":
  X = random.normal(2, 0.4, 500)
  a = min(X)
  b = max(X)
  F, bins= make_histogram(X, 100)
  P = normalized(F)
  D = DiscreteRandomVariable(X, P, bins)
  for i in range(10):
    cumulative(1., D, 1+0.2*i) 
