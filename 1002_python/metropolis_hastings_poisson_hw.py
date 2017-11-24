'''
aux_ind : sample x
n: iterations
'''
"""

"""
import os.path
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import factorial
from scipy.stats import poisson
import time

plt.close("all")

def mh_poisson_basic(n, lambda_par):
    bernoulli_samples = (np.random.rand(n) > 0.5).astype(int)
    unif_samples = np.random.rand(n)
    realization = np.zeros(n)
    x = 0
    for ind in range(1,n):
        if bernoulli_samples[ind] == 0:
            if x == 0:
                x = 0
            elif unif_samples[ind] <  np.true_divide( x, lambda_par):
                x = x - 1
        else:
            if unif_samples[ind] <  np.true_divide( lambda_par, x + 1):
                x = x + 1
        realization[ind] = x
    return realization

def mh_poisson_extended(n, lambda_par,m):
    realization = np.zeros(n)
    unif_samples1 = np.random.rand(n) 
    unif_samples2 = np.random.rand(n) 
    x = 0
    for ind in range(1, n):
        # k = j + 1
        if unif_samples1[ind] < 0.25:
            if unif_samples2[ind] < (lambda_par / (x+1)):
                x = x + 1
        elif unif_samples1[ind] < 0.5:
            if unif_samples2[ind] < (lambda_par ** m * factorial(x) / factorial(x + m)):
                x = x + m
        elif unif_samples1[ind] < 0.75:
            if x < m:
                pass
            else:
                if unif_samples2[ind] < ( factorial(x) / (lambda_par ** m * factorial(x - m))):
                    x = x - m
        else:
            if x == 0:
                pass
            else:
                if unif_samples2[ind] < (x / lambda_par):
                    x = x - 1
        realization[ind] = x
    return realization

n = 200
n_tries = 10000
m=5

# Poisson parameter
lambda_par = 40

# True pmf
aux_ind = np.arange(25,41,5)
poisson_pmf = poisson.pmf(aux_ind, lambda_par)
np.true_divide( np.exp(-lambda_par) * lambda_par ** aux_ind , factorial(aux_ind) )

# Empirical pmf obtained by Metropolis Hastings
empirical_pmf_basic_aux = np.zeros((len(aux_ind),n))
empirical_pmf_extended_aux = np.zeros((len(aux_ind),n))
start_time = time.time()
for i_try in range(n_tries):
    realization_basic = mh_poisson_basic(n,lambda_par)
    realization_extended = mh_poisson_extended(n,lambda_par,m)
    for ind in range(len(aux_ind)):
        empirical_pmf_basic_aux[ind,:] = empirical_pmf_basic_aux[ind,:] + (realization_basic == aux_ind[ind]).astype(int)
        empirical_pmf_extended_aux[ind,:] = empirical_pmf_extended_aux[ind,:] + (realization_extended == aux_ind[ind]).astype(int)
    empirical_pmf_basic = np.true_divide(empirical_pmf_basic_aux, n_tries)
    empirical_pmf_extended = np.true_divide(empirical_pmf_extended_aux, n_tries)
print("--- %s seconds ---" % (time.time() - start_time))


fig = plt.figure(figsize=(14, 7)) 
color_array = ['limegreen', 'purple', 'grey', 'steelblue', 'deepskyblue', 'tomato',
               'maroon','darkgray','darkorange', 'steelblue', 'forestgreen', 'silver']
for ind in range(len(aux_ind)):
    plt.plot([1,n],[poisson_pmf[ind],poisson_pmf[ind]],'--',lw=2, color=color_array[ind])
    plt.plot(np.arange(n),empirical_pmf_basic[ind,:],lw=2, color=color_array[ind], label=str(aux_ind[ind]))    
plt.legend(fontsize=18)
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18)
plt.ylabel("Distribution", fontsize=22,labelpad=10)  
plt.xlabel("Iterations", fontsize=22,labelpad=10)  
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('metropolis_hastings_poisson_basic.pdf')

fig = plt.figure(figsize=(14, 7)) 
for ind in range(len(aux_ind)):
    plt.plot([1,n],[poisson_pmf[ind],poisson_pmf[ind]],'--',lw=2, color=color_array[ind])
    plt.plot(np.arange(n),empirical_pmf_extended[ind,:],lw=2, color=color_array[ind], label=str(aux_ind[ind]))    
plt.legend(fontsize=18, loc=2)
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18)
plt.ylabel("Distribution", fontsize=22,labelpad=10)  
plt.xlabel("Iterations", fontsize=22,labelpad=10)  
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('metropolis_hastings_poisson_extended.pdf')
