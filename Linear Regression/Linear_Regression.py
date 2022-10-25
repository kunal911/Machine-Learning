#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import inv
import math
import statistics
import matplotlib.pyplot as plot


C_TRAINING = "./concrete/train.csv"
C_TESTING = "./concrete/test.csv"
with open(C_TRAINING,mode='r') as f:
    c_t=[]
    for line in f:
        t=line.strip().split(',') 
        c_t.append(t)
with open(C_TESTING,mode='r') as f:
    c_tt = []
    for line in f:
        t=line.strip().split(',') 
        c_tt.append(t)
def convert_to_float(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data
m = len(c_t)   
d = len(c_t[0]) - 1

c_t = convert_to_float(c_t)  
c_tt = convert_to_float(c_tt) 


def get_cost(w, ds):
    loss = 0.5*sum([ (row[-1]-np.inner(w,row[0:7]))**2 for row in ds ])
    return loss

def gradient(w, ds):
    gradient = []
    for j in range(d):
        gradient.append(-sum([ (row[-1]-np.inner(w, row[0:7]))*row[j] for row in ds]))
    return gradient

def batch_gradient(eps, r, w, ds):
    cost =[]
    while np.linalg.norm(gradient(w, ds)) >= eps:
        cost.append(get_cost(w, ds))
        w = w - [r*x for x in gradient(w, ds)]       
    return [w, cost]


def sgd_single(eps, r, w, ds, pi):
    flag = 0
    loss_vector =[]
    for x in pi:
        if np.linalg.norm(sgd_gradient(w, pi[x], ds)) <= eps:
            flag = 1
            return [w, loss_vector, flag]
        loss_vector.append(get_cost(w, ds))
        w = w - [r*x for x in sgd_gradient(w, pi[x] ,ds)]     
    return [w, loss_vector, flag]


def sgd_gradient(w, sample_idx, ds):
    s_grad = []
    for j in range(d):
        s_grad.append(-(ds[sample_idx][-1]-np.inner(w, ds[sample_idx][0:7]) )*ds[sample_idx][j])
    return s_grad

def sgd_random(eps, r, w, ds, N_epoch ):
    loss_all =[]
    for i in range(N_epoch):
        pi = np.random.permutation(m)
        [w, loss_vector, flag] = sgd_single(eps, r, w, ds, pi)
        if flag == 1:
            return [w, loss_all]
        loss_all = loss_all + loss_vector
    return [w, loss_all]



[ww, loss] = batch_gradient(0.0001, 0.01, np.zeros(d), c_t)
print(ww)
print(get_cost(ww, c_t))
print(get_cost(ww, c_tt))
plot.plot(loss)
plot.ylabel("cost value")
plot.xlabel("steps")
plot.title("Batch Gradient")
plot.show()


[ww, cost] = sgd_random(0.000001, 0.001, np.zeros(d), c_t, 20000)
print(ww)
print(get_cost(ww, c_t))
print(get_cost(ww, c_tt))
plot.plot(cost)
plot.ylabel("cost value")
plot.xlabel("steps")
plot.title("Stochastic Gradient")
plot.show()


d_list = [row[0:7] for row in c_t]
l_list = [row[-1] for row in c_tt]
d_mat = np.array(d_list)
l_mat = np.array(l_list)
X = d_mat.transpose()
a = inv(np.matmul(X, X.transpose()))
b = np.matmul(a, X)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




