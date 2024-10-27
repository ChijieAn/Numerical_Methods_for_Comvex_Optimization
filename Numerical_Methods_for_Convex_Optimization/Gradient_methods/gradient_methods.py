import autograd.numpy as np  # 使用autograd的numpy版本
from autograd import grad
from scipy.sparse import diags
from scipy.sparse import eye


#the grad method from last time, we compare with is as a baseline method
def gradmeth(fun, x0, tol, maxit):
    f_all = []
    gnorm_all = []
    x = x0
    alpha = 0.25  # initial step size, you may want to adjust this alpha in (0,0.5)
    beta = 0.5  # step size reduction factor, you may want to adjust this beta in (0,1)

    for _ in range(maxit):
        print(_)
        f0, g0 = fun(x)  # Get the function value and gradient at the current point
        gnorm = np.linalg.norm(g0)
        gnorm_all.append(gnorm)
        if gnorm < tol:  # Check the stopping criterion
            break

        t = 1
        while fun(x - t * g0)[0] > f0 - alpha * t * gnorm**2:  # Backtracking line search
            t *= beta

        x = x - t * g0  # Update the point
        #print('g0 is',g0)
        #print('the new x is',x)
        f_all.append(fun(x)[0])  # Evaluate function at the new point

    return f_all, gnorm_all

#optimal gradient method
def opt_grad_method(fun, x0, tol, maxit, m, M):
    k = M / m
    q = (1 - 1/np.sqrt(k)) / (1 + 1/np.sqrt(k))
    x = x0
    y = x0
    f_all = []
    gnorm_all = []

    for _ in range(maxit,):
        print('this is iteration',_)
        print('y is',y)
        _,grad_y = fun(y)
        print('gradient is',grad_y)
        f0,g0 = fun(x)
        gnorm = np.linalg.norm(g0)
        gnorm_all.append(gnorm)

        # 更新x和y
        print(y.shape,grad_y.shape)
        print('1/M is',1/M)
        x_new = y - (1/M) * grad_y
        y = x_new + q * (x_new - x)
        x = x_new
        print('x_new is',x_new)
        # 记录历史信息
        f_all.append(fun(x)[0])
    optimal_value, _ = fun(x)
    return f_all, gnorm_all, optimal_value

#the original gradient method with a fixed stepsize of 1/M
def grad_method_fixed1(fun, x0, tol, maxit, m, M ):
    f_all = []
    gnorm_all = []
    x = x0
    alpha = 0.25 #alpha is taken from 0-0.5, usually take 0.25
    beta = 0.5  # step size reduction factor, you may want to adjust this beta in (0,1)

    for _ in range(maxit):
        print(_)
        f0, g0 = fun(x)  # Get the function value and gradient at the current point
        if _==0:
          print('the gradient 0 is',g0)
        if _==1:
          print('x is',x)
          print('1/M is',1/M)
        gnorm = np.linalg.norm(g0)
        gnorm_all.append(gnorm)
        if gnorm < tol:  # Check the stopping criterion
            break

        t = 1/M
        while fun(x - t * g0)[0] > f0 - alpha * t * gnorm**2:  # Backtracking line search
            t *= beta

        x = x - t * g0  # Update the point
        #print('g0 is',g0)
        #print('the new x is',x)
        f_all.append(fun(x)[0])  # Evaluate function at the new point

    return f_all, gnorm_all

def grad_method_fixed2(fun, x0, tol, maxit, m, M):
    f_all = []
    gnorm_all = []
    x = x0
    alpha = 0.25 #alpha is taken from 0-0.5, usually take 0.25
    beta = 0.5  # step size reduction factor, you may want to adjust this beta in (0,1)

    for _ in range(maxit):
        print(_)
        f0, g0 = fun(x)  # Get the function value and gradient at the current point
        if _==0:
          print('the gradient 0 is',g0)
        if _==1:
          print('x is',x)
        gnorm = np.linalg.norm(g0)
        gnorm_all.append(gnorm)
        if gnorm < tol:  # Check the stopping criterion
            break

        t = 2/(M+m)
        while fun(x - t * g0)[0] > f0 - alpha * t * gnorm**2:  # Backtracking line search
            t *= beta

        x = x - t * g0  # Update the point
        #print('g0 is',g0)
        #print('the new x is',x)
        f_all.append(fun(x)[0])  # Evaluate function at the new point

    return f_all, gnorm_all

# The test case 1 same as HW 5
def quad(x, A, b):
    f = 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x)
    g = np.dot(A, x) + b
    return f, g

def max_min_eigenvalues(n):
    eigenvalues = np.linalg.eigvals(A)
    max_eigenvalue = np.max(eigenvalues)
    min_eigenvalue = np.min(eigenvalues)
    return max_eigenvalue, min_eigenvalue

def hessian(M, m, size):
  diagonals = [2] * size
  off_diagonals = [-1] * (size - 1)
  hessian_matrix = diags([off_diagonals, diagonals, off_diagonals], offsets=[-1, 0, 1], shape=(size, size))
  #print(hessian_matrix)
  #print(diags([np.ones(size)], offsets=[0], shape=(size, size)))
  hessian_matrix = (M - m) / 4 * hessian_matrix + m * diags([np.ones(size)], offsets=[0], shape=(size, size))
  return hessian_matrix

#define the function for test example 3, which is the worst case example
def worst_function(x, M, m):
  n = len(x)
  first_term = (M - m) / 8 * x[0]**2
  summation_term = (M - m) / 8 *sum((x[i] - x[i+1])**2 for i in range(n-1))
  last_term = (M - m) / 8 *(-2 * x[0]) + m / 2 * np.linalg.norm(x)**2
  value = first_term+summation_term+last_term

  grad = np.zeros_like(x)
  grad[0] = (M - m) / 8 * (4 * x[0] - 2 * x[1] - 2) + m * x[0]
  for j in range(1, len(x)):
    if j+1 < len(x):
      grad[j] = (M-m)/8*(4*x[j]-2*x[j+1]-2*x[j-1])+m*x[j]
    elif j+1 == len(x):
      grad[j] = (M-m)/8*(4*x[j]-2-2*x[j-1])+m*x[j]

  sparse_identity = eye(n, format='csr')
  hess =  hessian(M,m,n)

  return value,grad