import autograd.numpy as np  # 使用autograd的numpy版本
from autograd import grad
from autograd import hessian
import copy

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

def objective_function(A,x):
    criterion = True
    for i in range(len(A[0])):
        if np.dot(A[:,i],x) >1:
            criterion = False
            break
    for j in range(len(A)):
        if abs(x[j])>1:
            criterion = False
            break
    if criterion == False:
        value = np.inf
        gradient = np.full((n, 1), np.nan)
    else:
        fun = lambda A, x: -np.sum([np.log(1 - np.dot(A[:, i], x)) for i in range(A.shape[1])]) - np.sum(np.log(1 - x[i]**2) for i in range(len(x)))
        value= fun(A,x)
    #then we compute the gradient
        '''term1=np.sum([A[:, i]/(1-np.dot(A[:, i], x)) for i in range(A.shape[1])])
        col_vector = np.arange(len(x)).reshape(len(x), 1)
        for i in range(len(x)):
            #print(n)
            #print(i)
            #print(x[i])
            col_vector[i, 0] = (2*x[i]/1-x[i]**2)
        term2 = col_vector
        print('this is term 1',term1)
        print('this is term 2',term2)
        gradient = term1+term2
        print(value,gradient)'''
        grad_fun = grad(fun, 1)
        gradient = grad_fun(A, x)
    return value,gradient

def newtmeth(fun, x0, tol, maxit):
    x = x0
    x_lst = []
    f_all = []
    hessian = []
    gradient_norm = []

    for _ in range(maxit):
      print(_)
      f, grad_f, hessian_f = fun(x)
      f_all.append(f)
      hessian.append(hessian_f)
      print(x)
      value_x = copy.deepcopy(x)
      x_lst.append(value_x)
      gnorm = np.linalg.norm(grad_f)
      gradient_norm.append(gnorm)

      print(hessian_f.shape,grad_f.shape)

      delta_x_nt = -np.linalg.solve(hessian_f, grad_f)
      lambda_2 = np.dot(grad_f.T, np.linalg.solve(hessian_f, grad_f))

      if lambda_2 / 2 <= tol:
          break

      #conduct line search
      t = 1.0
      alpha = 0.25
      beta = 0.5

      while fun(x + t * delta_x_nt)[0] > f + alpha * t * lambda_2:
            t *= beta
      #print(t*delta_x_nt)
      x += t * delta_x_nt

    return x_lst, f_all, gradient_norm,  hessian

def objective_function2(A,x):
    criterion = True
    for i in range(len(A[0])):
        if np.dot(A[:,i],x) >1:
            criterion = False
            break
    for j in range(len(A)):
        if abs(x[j])>1:
            criterion = False
            break
    if criterion == False:
        value = np.inf
        gradient = np.full((n, 1), np.nan)
        hessian_mat = np.full((n,n),np.nan)

    else:
        fun = lambda A, x: -np.sum([np.log(1 - np.dot(A[:, i], x)) for i in range(A.shape[1])]) - np.sum(np.log(1 - x[i]**2) for i in range(len(x)))
        value= fun(A,x)
    #then we compute the gradient and hessian
        grad_fun = grad(fun, 1)
        gradient = grad_fun(A, x)
        hessian_fun = hessian(fun,1)
        hessian_mat = np.squeeze(hessian_fun(A, x))

    return value,gradient,hessian_mat