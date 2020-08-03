# ## Inference of Gaussian Mixture Models using Gibb's sampling

# Please refer to Gibb's sampling pdf attached in the same directory titled "LR_inference_via_gibbs_sampling" to get a summary of Gibb's sampling.
# Here, we will look at Gaussian Mixture Model (GMM) and the explanation and derivation of the equations are presented in the pdf titled "Gibbs_inference_GMM".

# ### GMMs

# Let us take the example of average height of male and female in different countries.
# I have uploaded a csv file titled "heights" in the same repository. Let us plot the data from this file.

import pandas as pd
import numpy as np
import random
import scipy.stats
import matplotlib.pyplot as plt

def sample_mus(data,z,tau,mu_0,tau_0):
    N     = len(data)
    assert len(z) == N
    
    N_k   = [N-sum(z),sum(z)]
    
    precision_f = tau_0[0] + tau*N_k[0]
    precision_m = tau_0[1] + tau*N_k[1]
    
    mean_f = ((tau_0[0]*mu_0[0]) + tau*sum([data[i] for i in range(N) if z[i]==0])) / precision_f
    mean_m = ((tau_0[1]*mu_0[1]) + tau*sum([data[i] for i in range(N) if z[i]==1])) / precision_m
    
    return [np.random.normal(mean_f, 1/np.sqrt(precision_f)),np.random.normal(mean_m, 1/np.sqrt(precision_m))],N_k


def sample_z(data,pi,mus,tau):
    
    x_minus_mu          = [(d - mus[0],d - mus[1]) for d in data]
    post_z_given_x      = [(pi[0]*scipy.stats.norm(0,1/np.sqrt(tau)).pdf(f), pi[1]*scipy.stats.norm(0,1/np.sqrt(tau)).pdf(m)) for (f,m) in x_minus_mu]
    norm_post_z_given_x = [(f/(f+m),m/(f+m)) for (f,m) in post_z_given_x]
    z                   = [0 if f > m else 1 for (f,m) in norm_post_z_given_x]

    return z


def sample_pi(N_k,alpha):
    return np.random.dirichlet(alpha + np.array(N_k))

# Sampling function
def gibbs_sampling(data,iters,hyper,z):
    assert len(data) == len(z)
    N = len(data)
    
    mus = [hyper["mu_f0"],hyper["mu_m0"]]
    
    trace = np.zeros((iters, 4))
    N_k = [N-sum(z),sum(z)]           
    
    for it in range(iters):
        pi         = sample_pi(N_k,hyper["alpha"])
        z          = sample_z(data,pi,mus,hyper["tau"])
        mus, N_k   = sample_mus(data,z,hyper["tau"],[hyper["mu_f0"],hyper["mu_m0"]],[hyper["tau_f0"],hyper["tau_m0"]])
        
        trace[it,:] = np.array(mus+list(pi))
        
    trace = pd.DataFrame(trace)
    trace.columns = ['mu_f', 'mu_m', 'pi_f', 'pi_m']
    return trace

def main():
    
    # Using pandas dataframe to read and store the csv file
    df_ht = pd.read_csv("~/heights.csv")


    # Replacing NaN with the mean of that column
    df_ht.fillna(df_ht.mean())


    # Plot a histogram of the data
    bins = np.linspace(140, 185, 90)

    plt.figure()
    plt.hist(df_ht.femaleMetricHeight, bins, alpha=0.75, label='female')
    plt.hist(df_ht.maleMetricHeight, bins, alpha=0.75, label='male')
    plt.legend()
    plt.xlabel("height (cm)")
    plt.grid()

    # # True parameter values of the female distribution
    # df_ht.femaleMetricHeight.mean(), df_ht.femaleMetricHeight.std()

    # # True parameter values of the male distribution
    # df_ht.maleMetricHeight.mean(), df_ht.maleMetricHeight.std()

    # Create a dataset with only heights from the dataframe and shuffle them
    data = list(df_ht.femaleMetricHeight)+list(df_ht.maleMetricHeight)
    random.shuffle(data)

    # Filter out outliers - the entires that are 0
    print("Length of the dataset before removing outliers:",len(data))
    data = list(filter(lambda d: d>=1e-2,data))
    print("Length of the dataset after removing outliers: ",len(data))


    # Find the mean and std dev. of the entire dataset
    mu_0    = sum(data)/len(data)
    sigma_0 = np.std(np.array(data))
    print("mean and std dev. of the entire dataset: ",mu_0,sigma_0)
    
    
    iters = 30
    
    # Specify hyper parameters
    hyper  = {"tau": 1/4,
              "mu_f0":150,
              "tau_f0":1/16,
              "mu_m0":200,
              "tau_m0":1/16,
              "alpha": 2,
              "K": 2}

    # Assign a random label to each height
    z = [random.randrange(0,2) for i in range(len(data))]

    # Run Gibb's sampling
    trace = gibbs_sampling(data,iters,hyper,z)


    # Plot the trace
    fig, axs = plt.subplots(2)
    axs[0].plot(trace.loc[:,["mu_f","mu_m"]])
    axs[0].legend(["mu_f","mu_m"])
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Parameter value")
    axs[0].grid()

    axs[1].plot(trace.loc[:,["pi_f","pi_m"]])
    axs[1].legend(["pi_f","pi_m"])
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Parameter value")
    axs[1].grid()
    plt.show()

    mu_f_burnin = trace.loc[20:,["mu_f"]].mean()[0]
    mu_m_burnin = trace.loc[20:,["mu_m"]].mean()[0]

    print("Average female height in the world: {:3.4f}".format(mu_f_burnin))
    print("Average male height in the world:   {:3.4f}".format(mu_m_burnin))


# So, thats's it! We have successfully inferred the means from the heights data.
main()
