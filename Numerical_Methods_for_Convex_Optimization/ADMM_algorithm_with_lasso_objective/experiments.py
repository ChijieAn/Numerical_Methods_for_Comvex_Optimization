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

from admm_algorithms import soft_thresholding, soft_thresholding_sparse, admm_lasso, admm_lasso_sparse

#define the objective function for lasso problem
def f(x, A, b):
    return 0.5 * np.linalg.norm(A @ x - b, 2)**2
def g(z, lambda_val):
    return lambda_val * np.linalg.norm(z, 1)
def lasso_objective(x, A, b, lambda_val):
    return f(x, A, b) + g(x, lambda_val)

  #check by a small example
np.random.seed(0)  # For reproducibility
m, n = 30, 10  # m samples, n features
A = np.random.randn(m, n)
b = np.random.randn(m)
lmbda = 1.0
rho = 1.0

# Solve the Lasso problem
x,value,value_list,p_residual_list,d_residual_list = admm_lasso(A, b, lmbda, rho)

# Display the result
print(x,value)

#use the CVX package to solve the problem
x = cp.Variable(A.shape[1])
f_x = 0.5 * cp.norm2(A @ x - b)**2
g_z = lmbda * cp.norm1(x)
problem = cp.Problem(cp.Minimize(f_x + g_z))
problem.solve()
print(x.value,problem.value)

p_residual_norms = [np.linalg.norm(vec, 2) for vec in p_residual_list]
d_residual_norms = [np.linalg.norm(vec, 2) for vec in d_residual_list]
dist_to_optimal = [values - value for values in value_list]

plt.semilogy(p_residual_norms)
plt.title('Log Plot of Primal Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.show()

plt.semilogy(d_residual_norms)
plt.title('Log Plot of Dual Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.show()

plt.semilogy(dist_to_optimal)
plt.title('Log Plot of Dual Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.show()

#first we fix lambda = 1 and try different values of rho
#we try values in [0.1,0.5,1,5,10] for rho
#then we also try values in [100,500,1000,5000,10000] for rho
x_r_1,value_r_1,value_list_r_1,p_residual_list_r_1,d_residual_list_r_1 = admm_lasso(A, b, 1, 0.1)
x_r_2,value_r_2,value_list_r_2,p_residual_list_r_2,d_residual_list_r_2 = admm_lasso(A, b, 1, 0.5)
x_r_3,value_r_3,value_list_r_3,p_residual_list_r_3,d_residual_list_r_3 = admm_lasso(A, b, 1, 1)
x_r_4,value_r_4,value_list_r_4,p_residual_list_r_4,d_residual_list_r_4 = admm_lasso(A, b, 1, 5)
x_r_5,value_r_5,value_list_r_5,p_residual_list_r_5,d_residual_list_r_5 = admm_lasso(A, b, 1, 10)

p_residual_norms_r_1 = [np.linalg.norm(vec, 2) for vec in p_residual_list_r_1]
d_residual_norms_r_1 = [np.linalg.norm(vec, 2) for vec in d_residual_list_r_1]
dist_to_optimal_r_1 = [values - value for values in value_list_r_1]
p_residual_norms_r_2 = [np.linalg.norm(vec, 2) for vec in p_residual_list_r_2]
d_residual_norms_r_2 = [np.linalg.norm(vec, 2) for vec in d_residual_list_r_2]
dist_to_optimal_r_2 = [values - value for values in value_list_r_2]
p_residual_norms_r_3 = [np.linalg.norm(vec, 2) for vec in p_residual_list_r_3]
d_residual_norms_r_3 = [np.linalg.norm(vec, 2) for vec in d_residual_list_r_3]
dist_to_optimal_r_3 = [values - value for values in value_list_r_3]
p_residual_norms_r_4 = [np.linalg.norm(vec, 2) for vec in p_residual_list_r_4]
d_residual_norms_r_4 = [np.linalg.norm(vec, 2) for vec in d_residual_list_r_4]
dist_to_optimal_r_4 = [values - value for values in value_list_r_4]
p_residual_norms_r_5 = [np.linalg.norm(vec, 2) for vec in p_residual_list_r_5]
d_residual_norms_r_5 = [np.linalg.norm(vec, 2) for vec in d_residual_list_r_5]
dist_to_optimal_r_5 = [values - value for values in value_list_r_5]

#plot all the lists for 5 different values of rho in the same plot for comparasion
plt.semilogy(p_residual_norms_r_1, label='rho = 0.1, lambda = 1.0')
plt.semilogy(p_residual_norms_r_2, label='rho = 0.5, lambda = 1.0')
plt.semilogy(p_residual_norms_r_3, label='rho = 1.0, lambda = 1.0')
plt.semilogy(p_residual_norms_r_4, label='rho = 5.0, lambda = 1.0')
plt.semilogy(p_residual_norms_r_5, label='rho = 10.0, lambda = 1.0')
plt.title('Log Plot of Primal Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()
plt.semilogy(d_residual_norms_r_1, label='rho = 0.1, lambda = 1.0')
plt.semilogy(d_residual_norms_r_2, label='rho = 0.5, lambda = 1.0')
plt.semilogy(d_residual_norms_r_3, label='rho = 1.0, lambda = 1.0')
plt.semilogy(d_residual_norms_r_4, label='rho = 5.0, lambda = 1.0')
plt.semilogy(d_residual_norms_r_5, label='rho = 10.0, lambda = 1.0')
plt.title('Log Plot of Dual Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()
plt.semilogy(dist_to_optimal_r_1, label='rho = 0.1, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_2, label='rho = 0.5, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_3, label='rho = 1.0, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_4, label='rho = 5.0, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_5, label='rho = 10.0, lambda = 1.0')
plt.title('Log Plot of Distance to Optimal Value')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()

#Then we fix rho = 1 and change values of lambda to see its effects
#we try values in [0.1,1,10,100,1000] for lambda
x_l_1,value_l_1,value_list_l_1,p_residual_list_l_1,d_residual_list_l_1 = admm_lasso(A, b, 0.1, 1)
x_l_2,value_l_2,value_list_l_2,p_residual_list_l_2,d_residual_list_l_2 = admm_lasso(A, b, 1, 1)
x_l_3,value_l_3,value_list_l_3,p_residual_list_l_3,d_residual_list_l_3 = admm_lasso(A, b, 10, 1)
x_l_4,value_l_4,value_list_l_4,p_residual_list_l_4,d_residual_list_l_4 = admm_lasso(A, b, 100, 1)
x_l_5,value_l_5,value_list_l_5,p_residual_list_l_5,d_residual_list_l_5 = admm_lasso(A, b, 1000, 1)

print('lambda = 0.1',x_l_1)
print('lambda = 1',x_l_2)
print('lambda = 10',x_l_3)
print('lambda = 100',x_l_4)
print('lambda = 1000',x_l_5)

p_residual_norms_l_1 = [np.linalg.norm(vec, 2) for vec in p_residual_list_l_1]
d_residual_norms_l_1 = [np.linalg.norm(vec, 2) for vec in d_residual_list_l_1]
dist_to_optimal_l_1 = [values - value for values in value_list_l_1]
p_residual_norms_l_2 = [np.linalg.norm(vec, 2) for vec in p_residual_list_l_2]
d_residual_norms_l_2 = [np.linalg.norm(vec, 2) for vec in d_residual_list_l_2]
dist_to_optimal_l_2 = [values - value for values in value_list_l_2]
p_residual_norms_l_3 = [np.linalg.norm(vec, 2) for vec in p_residual_list_l_3]
d_residual_norms_l_3 = [np.linalg.norm(vec, 2) for vec in d_residual_list_l_3]
dist_to_optimal_l_3 = [values - value for values in value_list_l_3]
p_residual_norms_l_4 = [np.linalg.norm(vec, 2) for vec in p_residual_list_l_4]
d_residual_norms_l_4 = [np.linalg.norm(vec, 2) for vec in d_residual_list_l_4]
dist_to_optimal_l_4 = [values - value for values in value_list_l_4]
p_residual_norms_l_5 = [np.linalg.norm(vec, 2) for vec in p_residual_list_l_5]
d_residual_norms_l_5 = [np.linalg.norm(vec, 2) for vec in d_residual_list_l_5]
dist_to_optimal_l_5 = [values - value for values in value_list_l_5]

#plot all the lists for 5 different values of lambda in the same plot for comparasion
plt.semilogy(p_residual_norms_l_1, label='rho = 1.0, lambda = 0.1')
plt.semilogy(p_residual_norms_l_2, label='rho = 1.0, lambda = 1.0')
plt.semilogy(p_residual_norms_l_3, label='rho = 1.0, lambda = 10.0')
plt.semilogy(p_residual_norms_l_4, label='rho = 1.0, lambda = 100.0')
plt.semilogy(p_residual_norms_l_5, label='rho = 1.0, lambda = 1000.0')
plt.title('Log Plot of Primal Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()
plt.semilogy(d_residual_norms_l_1, label='rho = 1.0, lambda = 0.1')
plt.semilogy(d_residual_norms_l_2, label='rho = 1.0, lambda = 1.0')
plt.semilogy(d_residual_norms_l_3, label='rho = 1.0, lambda = 10.0')
plt.semilogy(d_residual_norms_l_4, label='rho = 1.0, lambda = 100.0')
plt.semilogy(d_residual_norms_l_5, label='rho = 1.0, lambda = 1000.0')
plt.title('Log Plot of Dual Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()
plt.semilogy(dist_to_optimal_l_1, label='rho = 1.0, lambda = 0.1')
plt.semilogy(dist_to_optimal_l_2, label='rho = 1.0, lambda = 1.0')
plt.semilogy(dist_to_optimal_l_3, label='rho = 1.0, lambda = 10.0')
plt.semilogy(dist_to_optimal_l_4, label='rho = 1.0, lambda = 100.0')
plt.semilogy(dist_to_optimal_l_5, label='rho = 1.0, lambda = 1000.0')
plt.title('Log Plot of Distance to Optimal Value')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()

#to avoid the problem of out of memory, we also need to write the sparse version of lasso objective function
from scipy.sparse.linalg import norm as spnorm
from scipy.sparse import issparse, csr_matrix

def f_sparse(x, A, b):
  #print((A@x).shape)
  #print(b.shape)
  #print(A@x-b)
  Ax_dense = (A @ x).toarray()
  b_dense = b.toarray()
  #print(Ax_dense)
  #print(b_dense)
  return 0.5 * np.linalg.norm(Ax_dense - b_dense)**2

def g_sparse(z, lambda_val):
  return lambda_val * np.sum(np.abs(z))

def lasso_objective_sparse(x, A, b, lambda_val):
  print(f_sparse(x,A,b))
  print(g_sparse(x,lambda_val))
  return f_sparse(x, A, b) + g_sparse(x, lambda_val)



#then we try on a bigger example
M, N = 100000, 10000
density = 2/M
rho2 = 0.55
lmbd2 = 1.0
A2 = sp.rand(M, N, density=density, format='csr')
b2 = np.random.rand(M, 1)
b2 = csr_matrix(b2)
I2 = sp.eye(N, format='csr')
#take a look at the sparsity pattern of A
plt.spy(A2, markersize=0.1)
plt.title('Sparsity Pattern of Matrix A')
plt.show()
B = A2.T.dot(A2) + rho2 * I2
plt.spy(B, markersize=0.1)
plt.title('Sparsity Pattern of B')
plt.show()
#B_csc = B.to_csc()
L2 = splu(A2.T.dot(A2)+ rho2 * I2).L
plt.spy(L2,markersize=0.1)
plt.title('Sparsity Pattern of Matrix L')
plt.show()

def L2(rho):
  B = A2.T.dot(A2) + rho * I2
  L2 = splu(B).L
  return L2

#run the experiment with basic choice of rho and lmbda
x,value,value_list,p_residual_list,d_residual_list = admm_lasso_sparse(A2, b2, lmbd2, rho2, L2(rho2), max_iter=1000)
print(x,value)

p_residual_norms = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list]
d_residual_norms = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list]
dist_to_optimal = [abs(values - value) for values in value_list]

#plot the log plots of distance to the optimal value, norm of primal and dual residuals
plt.semilogy(p_residual_norms)
plt.title('Log Plot of Primal Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.show()

plt.semilogy(d_residual_norms)
plt.title('Log Plot of Dual Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.show()

plt.semilogy(dist_to_optimal)
plt.title('Log Plot of Distance To. Optimal Value')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.show()

#first we fix lambda = 1 and try different values of rho
#we try the values [0.001,0.01,0.1,0.2,0.3,0.4,0.5] for rho
x_r_1,value_r_1,value_list_r_1,p_residual_list_r_1,d_residual_list_r_1 = admm_lasso_sparse(A2, b2, 1.0, 0.001, L2(0.001), max_iter=1000)
x_r_2,value_r_2,value_list_r_2,p_residual_list_r_2,d_residual_list_r_2 = admm_lasso_sparse(A2, b2, 1.0, 0.01, L2(0.01), max_iter=1000)
x_r_3,value_r_3,value_list_r_3,p_residual_list_r_3,d_residual_list_r_3 = admm_lasso_sparse(A2, b2, 1.0, 0.1, L2(0.1), max_iter=1000)
x_r_4,value_r_4,value_list_r_4,p_residual_list_r_4,d_residual_list_r_4 = admm_lasso_sparse(A2, b2, 1.0, 0.2, L2(0.2), max_iter=1000)
x_r_5,value_r_5,value_list_r_5,p_residual_list_r_5,d_residual_list_r_5 = admm_lasso_sparse(A2, b2, 1.0, 0.3, L2(0.3), max_iter=1000)
x_r_6,value_r_6,value_list_r_6,p_residual_list_r_6,d_residual_list_r_6 = admm_lasso_sparse(A2, b2, 1.0, 0.4, L2(0.4), max_iter=1000)
x_r_7,value_r_7,value_list_r_7,p_residual_list_r_7,d_residual_list_r_7 = admm_lasso_sparse(A2, b2, 1.0, 0.5, L2(0.5), max_iter=1000)

p_residual_norms_r_1 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_r_1]
d_residual_norms_r_1 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_r_1]
dist_to_optimal_r_1 = [abs(values - value_r_1) for values in value_list_r_1]
p_residual_norms_r_2 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_r_2]
d_residual_norms_r_2 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_r_2]
dist_to_optimal_r_2 = [abs(values - value_r_2) for values in value_list_r_2]
p_residual_norms_r_3 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_r_3]
d_residual_norms_r_3 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_r_3]
dist_to_optimal_r_3 = [abs(values - value_r_3) for values in value_list_r_3]
p_residual_norms_r_4 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_r_4]
d_residual_norms_r_4 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_r_4]
dist_to_optimal_r_4 = [abs(values - value_r_4) for values in value_list_r_4]
p_residual_norms_r_5 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_r_5]
d_residual_norms_r_5 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_r_5]
dist_to_optimal_r_5 = [abs(values - value_r_5) for values in value_list_r_5]
p_residual_norms_r_6 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_r_6]
d_residual_norms_r_6 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_r_6]
dist_to_optimal_r_6 = [abs(values - value_r_6) for values in value_list_r_6]
p_residual_norms_r_7 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_r_7]
d_residual_norms_r_7 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_r_7]
dist_to_optimal_r_7 = [abs(values - value_r_7) for values in value_list_r_7]

#plot all the lists for 5 different values of lambda in the same plot for comparasion
plt.semilogy(p_residual_norms_r_1, label='rho = 0.001, lambda = 1.0')
plt.semilogy(p_residual_norms_r_2, label='rho = 0.01, lambda = 1.0')
plt.semilogy(p_residual_norms_r_3, label='rho = 0.1, lambda = 1.0')
plt.semilogy(p_residual_norms_r_4, label='rho = 0.2, lambda = 1.0')
plt.semilogy(p_residual_norms_r_5, label='rho = 0.3, lambda = 1.0')
plt.semilogy(p_residual_norms_r_6, label='rho = 0.4, lambda = 1.0')
plt.semilogy(p_residual_norms_r_7, label='rho = 0.5, lambda = 1.0')
plt.title('Log Plot of Primal Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()
plt.semilogy(d_residual_norms_r_1, label='rho = 0.001, lambda = 1.0')
plt.semilogy(d_residual_norms_r_2, label='rho = 0.01, lambda = 1.0')
plt.semilogy(d_residual_norms_r_3, label='rho = 0.1, lambda = 1.0')
plt.semilogy(d_residual_norms_r_4, label='rho = 0.2, lambda = 1.0')
plt.semilogy(d_residual_norms_r_5, label='rho = 0.3, lambda = 1.0')
plt.semilogy(d_residual_norms_r_6, label='rho = 0.4, lambda = 1.0')
plt.semilogy(d_residual_norms_r_7, label='rho = 0.5, lambda = 1.0')
plt.title('Log Plot of Dual Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()
plt.semilogy(dist_to_optimal_r_1, label='rho = 0.001, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_2, label='rho = 0.01, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_3, label='rho = 0.1, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_4, label='rho = 0.2, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_5, label='rho = 0.3, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_6, label='rho = 0.4, lambda = 1.0')
plt.semilogy(dist_to_optimal_r_7, label='rho = 0.5, lambda = 1.0')
plt.title('Log Plot of Distance to Optimal Value')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.ylim(1e-15, 1e4)
plt.legend()
plt.show()

#then we try different values of lambda to do the same experiment
#we choose the values of lambda as [0.1,1,10,100]
x_l_1,value_l_1,value_list_l_1,p_residual_list_l_1,d_residual_list_l_1 = admm_lasso_sparse(A2, b2, 0.1, 0.1, L2(0.1), max_iter=1000)
x_l_2,value_l_2,value_list_l_2,p_residual_list_l_2,d_residual_list_l_2 = admm_lasso_sparse(A2, b2, 0.5, 0.1, L2(0.1), max_iter=1000)
x_l_3,value_l_3,value_list_l_3,p_residual_list_l_3,d_residual_list_l_3 = admm_lasso_sparse(A2, b2, 1.0, 0.1, L2(0.1), max_iter=1000)
x_l_4,value_l_4,value_list_l_4,p_residual_list_l_4,d_residual_list_l_4 = admm_lasso_sparse(A2, b2, 100.0, 0.1, L2(0.1), max_iter=1000)

#the length of x_l_1.data, which is the non strictly zero terms in x_l_1, is 8657
#however, due to the result is derived numerically, we can consider the terms which are
#very small, such as the terms less than 1e-15 as zero
nonzero_elements1 = x_l_1.data
count1 = np.sum(nonzero_elements1 < 1e-15)
print('lambda = 0.1, the number of non zero elemtents is',8657-count1)
nonzero_elements2 = x_l_2.data
count2 = np.sum(nonzero_elements2 < 1e-15)
print('lambda = 0.5, the number of non zero elements is',8657-count2)
nonzero_elements3 = x_l_3.data
count3 = np.sum(nonzero_elements3 < 1e-15)
print('lambda = 1.0, the number of non zero elements is',8657-count3)
nonzero_elements4 = x_l_4.data
count4 = np.sum(nonzero_elements4 < 1e-15)
print('lambda = 100.0, the number of non zero elements is',8657-count4)

p_residual_norms_l_1 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_l_1]
d_residual_norms_l_1 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_l_1]
dist_to_optimal_l_1 = [values - value for values in value_list_l_1]
p_residual_norms_l_2 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_l_2]
d_residual_norms_l_2 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_l_2]
dist_to_optimal_l_2 = [values - value for values in value_list_l_2]
p_residual_norms_l_3 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_l_3]
d_residual_norms_l_3 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_l_3]
dist_to_optimal_l_3 = [values - value for values in value_list_l_3]
p_residual_norms_l_4 = [np.linalg.norm(vec.toarray(), 2) for vec in p_residual_list_l_4]
d_residual_norms_l_4 = [np.linalg.norm(vec.toarray(), 2) for vec in d_residual_list_l_4]
dist_to_optimal_l_4 = [values - value for values in value_list_l_4]

#plot all the lists for 5 different values of lambda in the same plot for comparasion
plt.semilogy(p_residual_norms_l_1, label='rho = 0.1, lambda = 0.1')
plt.semilogy(p_residual_norms_l_2, label='rho = 0.1, lambda = 1.0')
plt.semilogy(p_residual_norms_l_3, label='rho = 0.1, lambda = 10.0')
plt.semilogy(p_residual_norms_l_4, label='rho = 0.1, lambda = 100.0')
#plt.semilogy(p_residual_norms_l_5, label='rho = 0.1, lambda = 1000.0')
plt.title('Log Plot of Primal Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()
plt.semilogy(d_residual_norms_l_1, label='rho = 0.1, lambda = 0.1')
plt.semilogy(d_residual_norms_l_2, label='rho = 0.1, lambda = 1.0')
plt.semilogy(d_residual_norms_l_3, label='rho = 0.1, lambda = 10.0')
plt.semilogy(d_residual_norms_l_4, label='rho = 0.1, lambda = 100.0')
#plt.semilogy(d_residual_norms_l_5, label='rho = 0.1, lambda = 1000.0')
plt.title('Log Plot of Dual Residuals l2 Norms')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()
plt.semilogy(dist_to_optimal_l_1, label='rho = 0.1, lambda = 0.1')
plt.semilogy(dist_to_optimal_l_2, label='rho = 0.1, lambda = 1.0')
plt.semilogy(dist_to_optimal_l_3, label='rho = 0.1, lambda = 10.0')
plt.semilogy(dist_to_optimal_l_4, label='rho = 0.1, lambda = 100.0')
#plt.semilogy(dist_to_optimal_l_5, label='rho = 1.0, lambda = 1000.0')
plt.title('Log Plot of Distance to Optimal Value')
plt.xlabel('Index')
plt.ylabel('Log of Value')
plt.legend()
plt.show()

