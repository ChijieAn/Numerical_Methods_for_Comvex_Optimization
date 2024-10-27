import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from scipy.sparse import eye
import numpy as np
from scipy.sparse import csr_matrix

#define the funciton for soft threshold
def soft_thresholding(x, lambda_val):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)

#define the function of ADMM for lasso regression
def admm_lasso(A, b, lmbda, rho, max_iter=10000):
    # Pre-compute some values
    Atb = A.T @ b
    AtA = A.T @ A
    m, n = A.shape
    I = np.eye(n)
    #initialize the variables
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    value_list = []
    p_residual_list = []
    d_residual_list = []

    L = np.linalg.cholesky(AtA + rho * I)
    for k in range(max_iter):
      # x-update (using Cholesky factorization), not using calculating inverse
      q = Atb + rho * (z - u)
      x = np.linalg.solve(L.T, np.linalg.solve(L, q))
      #then do the z update
      z_old = z
      z = soft_thresholding(x+u,lmbda/rho)
      #then do the u update
      u+=(x-z)
      #then calculate the objective value
      value = lasso_objective(x,A,b,lmbda)
      value_list.append(value)
      #calculate primal and dual residual
      p_residual = x-z
      p_residual_list.append(p_residual)
      d_residual = rho*(-I)@(z-z_old)
      d_residual_list.append(d_residual)
      #check p_residual and d_residual as stopping criterions
      if np.linalg.norm(p_residual, 2) < 1e-15 and np.linalg.norm(d_residual, 2) < 1e-15:
        #print(k)
        print('The algorithm reaches stopping criterion after iteration',k)
        break
    return x,value,value_list,p_residual_list,d_residual_list

#define the sparse version of admm, with the chelosky decomposition L computed outside of the algorithm
def soft_thresholding_sparse(x, lambda_val):
    dense_x = x.toarray().ravel()
    return np.sign(dense_x) * np.maximum(np.abs(dense_x) - lambda_val, 0)

def admm_lasso_sparse(A, b, lmbda, rho, L, max_iter=1000):
    # Pre-compute some values
    Atb = A.T @ b
    dense_array = Atb.toarray().ravel()
    Atb = csr_matrix(Atb)
    #print(Atb.shape)
    #Atb = Atb.flatten()
    m, n = A.shape
    I = eye(n, format='csr')
    #I = np.eye(n)
    #initialize the variables
    x = np.zeros(n)
    x = csr_matrix(x).T
    z = np.zeros(n)
    z = csr_matrix(z).T
    u = np.zeros(n)
    u = csr_matrix(u).T
    #print('the shape of x is',x.shape)
    #print('the shape of z is',z.shape)
    #print('the shape of u is',u.shape)
    value_list = []
    p_residual_list = []
    d_residual_list = []
    value = 0

    for k in range(max_iter):
      print(k)
      # x-update (using Cholesky factorization), not using calculating inverse
      #print('the shape of Atb is',Atb.shape)
      #print('the shape of z-u',(z-u).shape)
      q = Atb + rho * (z - u)
      #print(q.shape)
      x=csr_matrix(spsolve(L.T,spsolve(L,q))).T
      #print(x.shape)
      #print(u.shape)
      #then do the z update
      z_old = z
      z = csr_matrix(soft_thresholding_sparse(x+u,lmbda/rho).T).T
      #print(x.shape)
      #print(z.shape)
      #then do the u update
      u+=(x-z)
      p_residual = x-z
      #print(p_residual.shape)
      p_residual_list.append(p_residual)
      d_residual = rho*(-I)@(z-z_old)
      #print(d_residual.shape)
      d_residual_list.append(d_residual)
      value = lasso_objective_sparse(x,A,b,lmbda)
      #print(value)
      value_list.append(value)
      '''if np.linalg.norm(p_residual, 2) < 1e-15 and np.linalg.norm(d_residual, 2) < 1e-15:
        #print(k)
        print('The algorithm reaches stopping criterion after iteration',k)
        break'''

    return x,value,value_list,p_residual_list,d_residual_list