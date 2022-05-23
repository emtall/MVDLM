import scipy, math
import numpy as np
from scipy.stats import wishart , norm, multivariate_t, multivariate_normal 
from scipy.stats._multivariate import multivariate_t_frozen


class multivariate_t_with_cdf(multivariate_t_frozen):
    
    def __init__(self,*args, **kwargs): 
        
        super(multivariate_t_with_cdf, self).__init__(*args, **kwargs)
        
    def __call__(self,*args, **kwargs): 
        
        super(multivariate_t_with_cdf, self).__call__(*args, **kwargs)
        
    
    def cdf(self, x):
        samps=self.rvs(size=100000)
        prob=np.mean(samps.T <= x.reshape(len(self.loc), 1))
        return(prob)

class DLM:
    
    def __init__(self, beta, delta, q, m0=None, c0=100, n0=20, D0=None):
        if m0 is None: m0=np.zeros(q)
        if D0 is None: D0=n0*np.eye(q)
        self.beta = beta
        self.delta = delta
        self.q = q
        self.m = m0
        self.m_store = []
        self.D = D0
        self.D_store = []
        self.c = c0
        self.c_store = []
        #self.n = n0
        #self.n_store = []
        #self.h = self.n + self.q -1
        #self.h_store = []
        self.h =1/(1-self.beta)
        self.h_store = []
        self.n = self.h-self.q+1
        self.n_store = []
        self.t = 0
        self.y = None
        
    def add_data(self, y, set_w = True):
        e = y-self.m
        r = self.c/self.delta
        q_t = r + 1
        A = r/q_t
        self.m = (self.m + A*e)
        self.m_store.append(self.m) 
        self.c = r - (A**2)*q_t
        self.c_store.append(self.c) 
        self.D = self.beta*self.D + np.reshape(e, (self.q,1))@np.reshape(e, (1,self.q))/q_t
        self.D_store.append(self.D) 
        self.h = self.beta*self.h + 1
        self.h_store.append(self.h) 
        self.n = self.h-self.q + 1
        self.n_store.append(self.n) 
        self.t = self.t +1
        if self.y is None:
            self.y = y
        else:
            self.y = np.row_stack([self.y, y])
        
        if set_w: self.set_w()
        
        
    def post_pred(self):
        
        return(multivariate_t_with_cdf(loc=self.m, shape=(self.c/self.delta+1)*self.D/self.n, df= self.beta*self.h, allow_singular=True))
        
    def backwards_sample(self):
    
        self.thetas = np.zeros((self.t, self.q))
        self.Sigmas = []
        Phi = wishart.rvs(self.n_store[-1]+self.q-1, self.D_store[-1])
        theta = multivariate_normal.rvs(self.m_store[-1], self.c_store[-1]*np.linalg.inv(Phi))
        self.thetas[-1] = theta
        self.Sigmas.append(np.linalg.inv(Phi))
        
        
        for t in range(self.t-2, -1, -1):
            m_star = (1-self.delta)*self.m_store[t] + self.delta*theta 
            c_star = (1-self.delta)*self.c_store[t]

            theta = multivariate_normal.rvs(m_star, c_star*self.Sigmas[-1])
            h = self.n_store[t]+self.q-1
            
            
            Phi = self.beta*Phi + wish_rv(int((1-self.beta)*h), np.linalg.inv(self.D_store[t]))

            self.thetas[t] = theta
            self.Sigmas.append(np.linalg.inv(Phi))
        
        self.Sigmas = np.flip(self.Sigmas, axis=0)
        
        
    def simulate(self):
        
        if not hasattr(self, 'Sigmas'):
            self.backwards_sample()
        
        self.sim_y=np.array([multivariate_normal(self.thetas[i], self.Sigmas[i]).rvs() for i in range(self.t)])
        
class AR_DLM:
    
    def __init__(self,  arp, beta, delta, q, data, weird_thing=False, risk_free=None):

        self.Y = np.array(data)
        self.beta = beta
        self.delta = delta
        self.q = q
        self.arp = arp
        self.p = 1+arp*q
        self.n = 3
        self.h = self.n+q-1
        self.D = self.h*np.eye(q)
        self.z = np.zeros((self.p,q))
        self.zq = np.zeros((q,1))
        M0 = self.z
        if self.p > 1:
            M0[1:q+1, :] = np.eye(q)
            M0[0, :] = self.Y[(arp):].mean(axis=0)*(1-.99)
        self.M = M0
        self.C = np.eye(self.p)
        self.data = data
        self.weird_thing = weird_thing
                           
                           
    def add_data(self, y, t, set_w=True):
        
        if t < self.arp:
            return
        
        self.y = y
        idx = list(range((t-1),(t-self.arp-1),-1))
        self.F = np.append(np.array([1]), np.ndarray.flatten(self.Y[idx]))
        self.f = self.M.T@self.F
        self.R = self.C/self.delta
        self.q_t = self.F.T@self.R@self.F + 1
        e = np.vstack(y-self.f)
        A = np.vstack(self.R@self.F/self.q_t)
        self.M = self.M + A@e.T
        self.C = self.R - A@A.T*self.q_t
        self.h = self.beta*self.h + 1
        self.n = self.h-self.q + 1
        self.D = self.beta*self.D + e@e.T/self.q_t
        self.St = self.D/self.n
        #self.St = (St + St.T)/2
        
        if self.p > 1:
            dm = np.minimum(1, np.diag(self.M[1:1+self.q, :self.q]))
            for i in range(self.q): self.M[1+i, i] = dm[i]
                
        idx = list(range(t,(t-self.arp),-1))
        self.F = np.append(np.array([1]), np.ndarray.flatten(self.Y[idx]))
        self.R = self.C/self.delta

        
        
    def post_pred(self):
        q_t = self.F.T@self.R@self.F + 1
        self.f_new = self.M.T@self.F
        
        if self.n >self.q/self.beta:
            D = self.beta*self.D
            h = self.h*self.beta
        else:
            D = self.D
            h = self.h
        
        n = (h-self.q+1)
        
        self.V = q_t*D/n* h/(h-2)   
         
        self.V = (self.V+self.V.T)/2
        
        return(multivariate_t_with_cdf(loc=self.f_new, shape=self.V*(h-2)/h , df=h))
        
    
