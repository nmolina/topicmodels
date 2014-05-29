

### INITIALIZATION AND INFERENCE
# Adapted from
# http://shuyo.wordpress.com/2011/06/27/collapsed-gibbs-sampling-estimation-for-latent-dirichlet-allocation-3/
from __future__ import division
import numpy as np

class LDA:
    def __init__(self, docs, K, V, alpha=.2, beta=.2):
        # docs : array of words (each word from 0..V-1)
        # K : number of topics
        # V : vocabulary size
        
        self.docs = docs
        # topics of words of documents
        self.z_d_n = []
        # word count of each document and topic
        self.n_d_z = np.zeros((len(self.docs), K)) + alpha
        # word count of each topic and term
        self.n_z_w = np.zeros((K, V)) + beta
        # word count of each document, topic, term
        # (for coupled LDA)
        self.n_d_z_w = np.zeros((len(self.docs), K, V)) + beta
        # word count of each topic
        self.n_z = np.zeros(K) + V * beta
        # words per document (for perplexity calc.)
        self.n_d = np.array([len(i) for i in self.docs]) + K * alpha
        self.N = sum([len(i) for i in self.docs])
        
        for d, doc in enumerate(docs):
            z_n = []
            for w in doc:
                z = np.random.randint(0, K)
                z_n.append(z)
                self.n_d_z[d, z] += 1
                self.n_z_w[z, w] += 1
                self.n_d_z_w[d, z, w] += 1
                self.n_z[z] += 1
            self.z_d_n.append(np.array(z_n))

    def infer(self):
        # Perform an iteration of Gibbs sampling for LDA
        for d, doc in enumerate(self.docs):
            z_n = []
            for n, w in enumerate(doc):
                z = self.z_d_n[d][n]
                self.n_d_z[d, z] -= 1
                self.n_z_w[z, w] -= 1
                self.n_z[z] -= 1
                
                # draw from the posterior
                p_z = self.n_z_w[:, w] * self.n_d_z[d] / self.n_z
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                
                self.z_d_n[d][n] = new_z
                self.n_d_z[d, new_z] += 1
                self.n_z_w[new_z, w] += 1
                self.n_z[new_z] += 1
            
    def infer2(self, weight=0.9):
        # Perform an iteration of Gibbs sampling for coupled LDA
        for d, doc in enumerate(self.docs):
            z_n = []
            for n, w in enumerate(doc):
                z = self.z_d_n[d][n]
                self.n_d_z[d, z] -= 1
                self.n_z_w[z, w] -= 1
                self.n_z[z] -= 1
                self.n_d_z_w[d, z, w] -= 1
            
                # draw from the posterior
                val = weight*self.n_d_z_w[d, :, w] + (1-weight)*self.n_z_w[:, w]
                p_z = val * self.n_d_z[d] / (weight*self.n_d_z[d, z] + (1-weight)*self.n_z[z])
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_d_n[d][n] = new_z
                self.n_d_z[d, new_z] += 1
                self.n_z_w[new_z, w] += 1
                self.n_z[new_z] += 1
                self.n_d_z_w[d, new_z, w] += 1
    
    def freq_over_time(self):
        ans = []
        for d in range(0, len(self.docs)-1):
            ans.append(sym_KL(self.n_d_z[d, :]/self.n_d[d],
                              self.n_d_z[d+1, :]/self.n_d[d+1]))
        return ans
    
    def top_over_time(self):
        ans = []
        #phi[d, z, w] = self.n_d_z_w[d, z, w]/self.n_d_z[z]
        for d in range(0, len(self.docs)-1):
                cur = 0
                for z in range(len(self.n_z)):
                    cur += sym_KL(self.n_d_z_w[d, z, :]/self.n_d_z[d, z],
                                  self.n_d_z_w[d+1, z, :]/self.n_d_z[d, z])
                ans.append(cur/len(self.n_z))
        return ans
            
    def perplexity(self):
        #phi[z,w] = self.n_z_w[z, w]/self.n_z[z]
        #theta[d,z] = self.n_d_z[d, z]/self.n_d[d]
        return np.exp(-1/self.N*(sum([sum([
            np.log(sum(self.n_z_w[:,w]/self.n_z*self.n_d_z[d,:]/self.n_d[d]))
            for w in doc]) for d,doc in enumerate(self.docs)])))

def sym_KL(p,q):
    return (KL(p, q) + KL(q, p))/2

def KL(p, q):
    # Kullback-Leibler divergence D(P || Q) for discrete distributions
    # p, q : array-like, dtype=float
    # Discrete probability distributions.
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
 
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def iterate(lda, runs=5, perp=[]):
    for i in range(runs):
        lda.infer()
        perplexity = lda.perplexity()
        print perplexity
        perp.append(perplexity)
        
def iterate2(lda, runs=5, perp=[]):
    for i in range(runs):
        lda.infer2()
        perplexity = lda.perplexity()
        print perplexity
        perp.append(perplexity)
        
def smooth(list,strippedXs=False,degree=10):  
    if strippedXs==True: return Xs[0:-(len(list)-(len(list)-degree+1))]  
    smoothed=[0]*(len(list)-degree+1)  
    for i in range(len(smoothed)):  
        smoothed[i]=sum(list[i:i+degree])/float(degree)
    return np.array(smoothed)



