"""Appendix Foreword:
All code was written by Joe Wilson, based on the papers cited. Data was manually transferred to import for a single pair of models. 
"""




import math as m
from statistics import median
import random as r
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Calculate the euclidian distance between points x, y. Where x and y are lists
# of length dim
def dist(x,y):
    s = 0
    for i in range(0,len(x)):
        s = (x[i]-y[i])**2
    return m.sqrt(s)
# =============================================================================
# =============================== Read Data ===================================
# =============================================================================

# Open file
f= open("Test_Data/exported_data.csv", "r")
#f= open("Test_Data/Sample_Data_6.csv", "r")
lines = f.readlines()

#Determine dimension of data
dim = int(lines[0].strip().split("_")[-1])+1
print("Dimension of data: %d" % dim)

#Determine number of samples
n_samples = int(lines[0].strip().split(",")[-1].split("_")[1].split("-")[0])+1
print("Number of Samples: %d" % n_samples)

#Make empty lists
samples = []
for i in range(0,n_samples):
    samples.append([])

# Fill empty lists
for line in lines[1:]:
    templine = line.strip().split(",")
    for i in range (0,n_samples):
        if(templine[dim*i] == "nan"):
            continue

        point = []
        for d in range(0,dim):
            point.append(float(templine[dim*i+d]))
        samples[i].append(point)

#Determine Size of samples
ns = []
n = 0
for i in range(0,n_samples):
    ns.append(len(samples[i]))
    n += len(samples[i])

#Make the pooled sample
pooled = []
for i in range(0,n_samples):
    pooled = pooled + samples[i]

# =============================================================================
# ==================  Test Equality of Samples using MMD  =====================
# =============================================================================
#This component is based off of the paper [1] in the referenes, licenced under
# CC By 4.0.

# First we must pick a Kernel which leads to a RKHS and satisfies the C3
# condition in the paper. I'm lazy so I'm going to use the Gaussian Radial
# Basis Function (RBF) chosen by the paper. Following the advice of the paper
# we let sigma be "the median distance between observed vectors in the pooled
# sample"

#Caclulate sigma (I'm pretty sure this has terrible time complexity but oh well)
distances = []
for i in range(0,n):
    for j in range(0,n):
        if (i != j):
            distances.append(dist(pooled[i],pooled[j]))
sigma = median(distances)
print("Sigma: %f" % sigma)

#Define kernel
def k(y,y_prime):
    return(m.exp((-1*dist(y,y_prime))/(2*sigma*sigma)))

# Next we want a function to calculate the inner products of the canonical 
# feature maps in the RKHS. Luckily [1] gives a helpful kernel trick that makes
# this way less painful.
def v(alpha,beta):
    result = 0
    for i in range(0,ns[alpha]):
        for j in range(0,ns[beta]):
            result += k(samples[alpha][i],samples[beta][j])
    return((1/(ns[alpha]*ns[beta]))*result)

# Now it is easy to calculate the test statistic using equation (8) from [1]
t_obs = 0
for i in range(0,n_samples):
    for j in range(0,n_samples):
        if(i < j):
            t_obs += ((ns[i]*ns[j])/n)*(v(i,i)+v(j,j)-2*v(i,j))
print("Test statistic: %f" % t_obs)

# =============================================================================
# ==================  Method 2: Random Permutation Method =====================
# =============================================================================
# The null distribution of out test statistic sucks (as in, it is unknown
# [at least as far as I know] but the paper [1] shows its asymtopic behaviour).
# Regardless, the paper proposes a random permutation method to estimate the
# p-value.

# Alter V so it considers the permuted samples instead
def v_star(alpha,beta,l):
    result = 0
    # Determine the index where sample alpha starts
    alpha_index = 0
    for i in range(0,alpha):
        alpha_index += ns[i]
    # Determine the index where sample beta starts
    beta_index = 0
    for i in range(0,beta):
        beta_index += ns[i]
    #print(alpha_index,beta_index)
    # Calculate sum
    for i in range(0,ns[alpha]):
        for j in range(0,ns[beta]):
            result += k(pooled[l[alpha_index+i]],pooled[l[beta_index+j]])
    # Return result
    return((1/(ns[alpha]*ns[beta]))*result)

# Calculate test statsistic for the a sample permuted by l
def t_star(l):
    result = 0
    for i in range(0,n_samples):
        for j in range(0,n_samples):
            if (i<j):
                result += ((ns[i]*ns[j])/n)*(v_star(i,i,l)+v_star(j,j,l)-2*v_star(i,j,l))
    return result

# =============================================================================
# =============================  Energy Method ================================
# =============================================================================
# Before conducting the actual test bit of the Random Permutation Method
# proposed in [1] we note that another test (outlined in reference [2]) relies
# on a very similar method which allows us to conduct the tests simultaniously.
# This one uses an "energy" function

# Find the starting indicies of each sample in the pooled sample.
starting_indicies = [0]*n_samples
for i in range(1,n_samples):
    starting_indicies[i]=starting_indicies[i-1]+ns[i-1]

#Define the energy function, defined in equation (6) of [2]
def energy(l):
    result = 0
    for i in range(0,n_samples):
        for j in range(0,n_samples):
            if(i<j):
                result += little_energy(i,j,l)
    return(result)

#Define the energy between two samples i and j, defined in equation (5) of [2]
def little_energy(i,j,l):
    term1 = 0
    term2 = 0
    term3 = 0
    for p in range(starting_indicies[i],starting_indicies[i]+ns[i]):
        for q in range(starting_indicies[j],starting_indicies[j]+ns[j]):
            term1 += dist(pooled[l[p]],pooled[l[q]])
        for p1 in range(starting_indicies[i],starting_indicies[i]+ns[i]):
            term2 += dist(pooled[l[p]],pooled[l[p1]])
    
    for p in range(starting_indicies[j],starting_indicies[j]+ns[j]):
        for q in range(starting_indicies[j],starting_indicies[j]+ns[j]):
            term3 += dist(pooled[l[p]],pooled[l[q]])
    term1 = (2/(ns[i]*ns[j]))*term1
    term2 = (1/(ns[i]**2))*term2
    term3 = (1/(ns[j]**2))*term3
    return(((ns[i]*ns[j])/(ns[i]+ns[j]))*(term1-term2-term3))

# =============================================================================
# ==========================  Random Permutations =============================
# =============================================================================
# Now for some large number N, we randomly permute the samples N times and for
# each permutation we calculate what the test statistic would've been if that
# was our original data set. The proportion of permutations with test statistics
# greater than our observed statistic is an apporximation of the p-value.
# Pick N
N = 500
# Set up permutation
l = [0]*n
for i in range(0,n):
    l[i]=i

# energy of observed sample
e_obs = energy(l)

# Track number of bootleged test statistics exceeding observed test statistic
exceed1 = 0
exceed2 = 0
# Loop over N random permutations
for i in range(0,N):
    #randomise permutation
    r.shuffle(l)
    temp_t_star = t_star(l)
    temp_energy = energy(l)
    # Test if permuted TS exceeds observed TS
    if (temp_t_star>t_obs):
        exceed1 += 1
    if (temp_energy > e_obs):
        exceed2 += 1
    #Print stuff out
    print("[1]: Permutation %d of %d gives approximated p-value %f" % ((i+1), (N), ((1/(i+1))*exceed1)))
    print("[2]: Permutation %d of %d gives approximated p-value %f" % ((i+1), (N), ((1/(i+1))*exceed2)))
#Determine approximated p-value
print("RP method approximate p-value: %f" %((1/N)*exceed1))
print("Energy method approximate p-value: %f" %((1/N)*exceed2))
approx_RP_p_value = round(((1/N)*exceed1),3)
approx_Energy_p_value = round(((1/N)*exceed2),3)
# # =============================================================================
# # ==================  Method 3: Welch-Satterthwaite chi^2 =====================
# # =============================================================================
# # Another method to approximate a p-value is suggested in the paper where the 
# # null distribution of the test statistic is assumbed to be a linear multiple of
# # a chi squared distribution with the same mean and variance.

# # We begin with defining the function k_tilde as in equation (27) of [1].
# # Noticing that the final term is a constant, even still I think this has n^3
# # time complexity which isn't great (but don't worry it's about to get worse).
# term4 = 0
# for u in range(0,n):
#     for v in range(0,n):
#         term4 += k(pooled[u],pooled[v])
# term4 = (1/(n*n))*term4
# def k_tilde(i,j):
#     term2 = 0
#     term3 = 0
#     for u in range(0,n):
#         term2 += k(i,pooled[u])
#         term3 += k(pooled[u],j)
#     return(k(i,j)-(1/n)*(term2+term3)+term4)

# # Now we turn to calculating the quantities at the top of page 10, starting with
# # $\hat{E}[\tilde{K}(y,y)]$. I think we're up to n^4 time now
# p10_1 = 0
# for i in range(0,n):
#     p10_1 += k_tilde(pooled[i],pooled[i])
# p10_1 = (1/n)*p10_1
# print("Page 10, part 1: %f" % p10_1)

# # Now the second quantity at the top of page 10, $\hat{Var}[\tilde{K}(y,y)]$.
# # This one also has time complexity n^4. There's a chance I could've calculated
# # this concurrently with the previous but I'm too scared.
# p10_2 = 0
# for i in range(0,n):
#     p10_2 += (k_tilde(pooled[i],pooled[i])-p10_1)**2
# p10_2 = (1/(n-1))*p10_2
# print("Page 10, part 2: %f" % p10_2)

# # Now for the third quantity on top of page 10 $\hat{E}[\tilde{K}^2(y,y')]. This
# # one is nice and fast at n^5 time complexity.
# p10_3 = 0
# for i in range(0,n):
#     if (i % 5 == 0):
#         print("%d of %d terms sumed" % (i,n))
#     for j in range(0,n):
#         if (i<j):
#             p10_3 += k_tilde(pooled[i],pooled[j])**2
# p10_3= ((2)/(n*(n-1)))*p10_3
# print("Page 10, part 3: %f" % p10_3)

# # Now we have what we need to calculate the terms on page 9
# eT = (n_samples-1)*p10_1
# ksum = 0
# for i in range(0,n_samples):
#     ksum += ((n-ns[i])**2)/((n**2)*ns[i])
# varT=ksum*p10_2+2*((n_samples-1)-ksum)*p10_3
# # Caclulation of Beta
# beta = varT/(2*eT)
# print("Beta: %f" % beta)
# # Calculation of d
# d1 = (2*(eT**2))/varT
# print("d: %f" % d1)

# # Summary output for this method.
# print("Test statistic: %f" % t_obs)
# print("Critical value: %f" % (beta*chi2.ppf(0.95,d1)))
# print("P-value: %f" % (chi2.sf(t_obs/beta,d1)))
# approx_WS_p_value = round(chi2.sf(t_obs/beta,d1),3)
# =============================================================================
# ============================== Scatter Plot =================================
# =============================================================================
xs = []
ys = []
for i in range(0,n_samples):
    temp_xs = []
    temp_ys = []
    for j in range(0,ns[i]):
        temp_xs.append(samples[i][j][0])
        temp_ys.append(samples[i][j][1])
    xs.append(temp_xs)
    ys.append(temp_ys)

plt.figure(figsize=(6,6))
plt.scatter(xs[0], ys[0], c='blue', label='Keras MDN', alpha=0.25)
plt.scatter(xs[1], ys[1], c='green', label='TFLite MDN', alpha=0.25)
plt.legend()
plt.text(1, 0.05, "RP: p-value: "+str(approx_RP_p_value),
         transform=plt.gca().transAxes, fontsize=12, color='black',
         horizontalalignment='right', verticalalignment='bottom')
# plt.text(1, 0.10, "WS: p-value: "+str(approx_WS_p_value),
#          transform=plt.gca().transAxes, fontsize=12, color='black',
#          horizontalalignment='right', verticalalignment='bottom')
plt.text(1, 0, "Energy: p-value: "+str(approx_Energy_p_value),
         transform=plt.gca().transAxes, fontsize=12, color='black',
         horizontalalignment='right', verticalalignment='bottom')
plt.show()
# =============================================================================
# =============================== References ==================================
# =============================================================================
# [1]: Ong, Z. P., Chen, A. A., Zhu, T., & Zhang, J.-T. (2023). Testing Equality
# of Several Distributions at High Dimensions: A Maximum-Mean-Discrepancy-Based 
# Approach. Mathematics (Basel), 11(20), 4374-.
# https://doi.org/10.3390/math11204374
#
#
# [2]: Szekely, G. J., & Rizzo, M. L. (2004, October 30). Testing for Equal Distributions in
# High Dimensions. https://web.archive.org/web/20110805230307/http://personal.bgsu.edu/~mrizzo/energy/reprint-ksamples.pdf