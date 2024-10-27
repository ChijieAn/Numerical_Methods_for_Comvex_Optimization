import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import random
from data import data_str1,data_str2,data_str3

def nuclear_norm_minimization(X, known_indices, tol=1e-5):
    X_star = cp.Variable(shape=X.shape)

    objective = cp.Minimize(cp.normNuc(X_star))

    constraints = [X_star[idx[0],idx[1]] == X[idx[0],idx[1]] for idx in known_indices]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    return X_star.value

#build the example matrices W1, W2, W3

data_rows = data_str1.strip().split("\n")
data = [row.split("\t") for row in data_rows]
W1 = np.array(data, dtype=float)
#print(W1)
print(W1.shape)

data_rows = data_str2.strip().split("\n")
data = [row.split("\t") for row in data_rows]
W2 = np.array(data, dtype=float)
#print(W1)
print(W2.shape)

data_rows = data_str3.strip().split("\n")
data = [row.split("\t") for row in data_rows]
W3 = np.array(data, dtype=float)
#print(W1)
print(W3.shape)

def main(matrix):
  know_indices1 = []
  error1 = []
  for k in range(1, len(matrix)*len(matrix[0])):
      i, j = np.random.randint(len(matrix)), np.random.randint(len(matrix[0]))
      while (i,j) in know_indices1:
          i, j = np.random.randint(len(matrix)), np.random.randint(len(matrix[0]))
      know_indices1.append((i,j))
      W_star = nuclear_norm_minimization(matrix, know_indices1)
      error = np.linalg.norm(matrix - W_star, 'fro')
      error1.append(error)
      if error < 1e-5:
          break
  print('the threshold is reached after iteration',k)
  return error1,k

error1,k = main(W1)
error2,k = main(W2)
error3,k = main(W3)

plt.semilogy(range(1, k+1), error1)
plt.xlabel('k')
plt.ylabel('Error (Frobenius norm)')
plt.show()

plt.semilogy(range(1, k+1), error2)
plt.xlabel('k')
plt.ylabel('Error (Frobenius norm)')
plt.show()

plt.semilogy(range(1, k+1), error3)
plt.xlabel('k')
plt.ylabel('Error (Frobenius norm)')
plt.show()

#then we generate more succesive sequences of indices, and plot all the results in one single plot
error11,k1 = main(W1)
error12,k2 = main(W1)
error13,k3 = main(W1)
error14,k4 = main(W1)
error15,k5 = main(W1)

plt.semilogy(range(1, k1+1), error11)
plt.semilogy(range(1, k2+1), error12)
plt.semilogy(range(1, k3+1), error13)
plt.semilogy(range(1, k4+1), error14)
plt.semilogy(range(1, k5+1), error15)
plt.legend(['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5'])
plt.xlabel('k')
plt.ylabel('Error (Frobenius norm)')
plt.show()

error21,k1 = main(W2)
error22,k2 = main(W2)
error23,k3 = main(W2)
error24,k4 = main(W2)
error25,k5 = main(W2)

plt.semilogy(range(1, k1+1), error21)
plt.semilogy(range(1, k2+1), error22)
plt.semilogy(range(1, k3+1), error23)
plt.semilogy(range(1, k4+1), error24)
plt.semilogy(range(1, k5+1), error25)
plt.legend(['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5'])
plt.xlabel('k')
plt.ylabel('Error (Frobenius norm)')
plt.show()

error31,k1 = main(W3)
error32,k2 = main(W3)
error33,k3 = main(W3)
error34,k4 = main(W3)
error35,k5 = main(W3)

plt.semilogy(range(1, k1+1), error31)
plt.semilogy(range(1, k2+1), error32)
plt.semilogy(range(1, k3+1), error33)
plt.semilogy(range(1, k4+1), error34)
plt.semilogy(range(1, k5+1), error35)
plt.legend(['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5'])
plt.xlabel('k')
plt.ylabel('Error (Frobenius norm)')
plt.show()