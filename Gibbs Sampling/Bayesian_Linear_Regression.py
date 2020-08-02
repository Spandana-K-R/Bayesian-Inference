# Gibb's Sampling for Linear Regression
# Refer to the attached PDF titled "LR_inference_via_gibbs_sampling" for explanation and derivation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sample_beta_0(y,x,beta_1,tau,mu_0,tau_0):
    N = len(y)
    assert len(x) == N
    
    precision = tau_0 + tau*N
    mean      = (tau_0*mu_0 + tau*np.sum(y - beta_1*x)) / precision
    
    return np.random.normal(mean, 1/np.sqrt(precision))


def sample_beta_1(y,x,beta_0,tau,mu_1,tau_1):
    N = len(y)
    assert len(x) == N
    
    precision = tau_1 + tau*np.sum(x*x)
    mean      = (tau_1*mu_1 + tau*np.sum(x*(y - beta_0))) / precision
    
    return np.random.normal(mean, 1/np.sqrt(precision))


def sample_tau(y,x,beta_0,beta_1,alpha,beta):
    N = len(y)
    assert len(x) == N
    
    alpha_new = alpha + N/2
    beta_new  = beta + np.sum((y - beta_0 - beta_1*x)**2)/2
    
    return np.random.gamma(alpha_new,1/beta_new)

def gibbs(y, x, iters, init, hypers):
    assert len(y) == len(x)
    beta_0 = init["beta_0"]
    beta_1 = init["beta_1"]
    tau = init["tau"]
    
    store_param = np.zeros((iters, 3))
    
    for i in range(iters):
        
        # Update beta_0 with other old parameters
        beta_0 = sample_beta_0(y, x, beta_1, tau, hypers["mu_0"], hypers["tau_0"])
        
        # Update beta_1 with new beta_0 and old tau
        beta_1 = sample_beta_1(y, x, beta_0, tau, hypers["mu_1"], hypers["tau_1"])
        
        # Finally update tau with new beta_0 and new beta_1
        tau    = sample_tau(y, x, beta_0, beta_1, hypers["alpha"], hypers["beta"])
        
        # Store the parameters sampled at every iteration
        store_param[i,:] = np.array((beta_0, beta_1, tau))
        
    store_param = pd.DataFrame(store_param)
    store_param.columns = ['beta_0', 'beta_1', 'tau']
        
    return store_param

def main():
    
    # Set up a toy example
    beta_0_true = -2
    beta_1_true = 5
    tau_true    = 3

    N = 50
    x = np.random.uniform(0,4,N)
    # y = beta_0 + beta_1*x
    y = np.random.normal(beta_0_true+beta_1_true*x,1/np.sqrt(tau_true))

    plt.figure()
    synth_plot = plt.plot(x,y,"o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()


    # Specify initial values
    init = {"beta_0": 0,
            "beta_1": 0,
            "tau": 2}

    # Specify hyper parameters
    hypers = {"mu_0": 0,
             "tau_0": 1,
             "mu_1": 0,
             "tau_1": 1,
             "alpha": 2,
             "beta": 1}

    iters = 500

    # Run the gibb's sampler for 500 iterations and get the trace of the parameters sampled
    store_param = gibbs(y, x, iters, init, hypers)

    # Plot the trace
    param_plot = store_param.plot()
    param_plot.set_xlabel("Iteration")
    param_plot.set_ylabel("Parameter value")
    param_plot.grid()

    # Print the mean and standard deviation of the posterior parameters
    print("Parameter means")
    print(store_param[:-200].median())

    print()

    print("Parameter standard deviations")
    print(store_param[:-200].std())

    # Histograms of the parameter estimations after the burn-in period
    hist_plot = store_param[:-200].hist(bins = 30, layout = (1,3))
    plt.show()

    
main()
