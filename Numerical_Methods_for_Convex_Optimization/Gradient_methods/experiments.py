import autograd.numpy as np  # 使用autograd的numpy版本
from autograd import grad
import matplotlib.pyplot as plt
from scipy.sparse import eye
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

from gradient_methods import grad_method_fixed1, grad_method_fixed2, gradmeth, opt_grad_method, quad, max_min_eigenvalues,hessian,worst_function
from data import data_str
n = 5
A = np.array([[1/(i+j-1) for i in range(1, n+1)] for j in range(1, n+1)])
b = np.ones((n, 1))
fun = lambda x: quad(x, A, b)
x0 = np.ones((n, 1))
tol = 1e-6
maxit = 10000
M,m = max_min_eigenvalues(5)
print(M,m)

f_all1, gnorm_all1 = grad_method_fixed1(fun, x0, tol, maxit,m,M)
f_all2, gnorm_all2 = grad_method_fixed2(fun,x0,tol,maxit,m,M)
f_all3, gnorm_all3,_ = opt_grad_method(fun,x0,tol,maxit,m,M)

A_inv = np.linalg.inv(A)
x = -np.dot(A_inv, b)
print('the minimizer x is',x)
p_star = fun(x)[0][0][0]
print('the optimal value is',p_star)

p_star1=f_all1[-1][0]
values_to_plot1 = f_all1 - p_star1
plt.semilogy(values_to_plot1.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with t=1/M')  # 图表标题
plt.show()

plt.semilogy(gnorm_all1, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for Gradient Method with t=1/M')  # 图表标题
plt.show()

p_star2=f_all2[-1][0]
values_to_plot2 = f_all2 - p_star2
plt.semilogy(values_to_plot2.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with t=2/(M+m)')  # 图表标题
plt.show()

plt.semilogy(gnorm_all2, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for gradient method when t=2/(m+M)')  # 图表标题
plt.show()

p_star3=f_all3[-1][0]
values_to_plot3 = f_all3 - p_star3
plt.semilogy(values_to_plot3.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with optimal gradient method')  # 图表标题
plt.show()

plt.semilogy(gnorm_all3, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for optimal gradient')  # 图表标题
plt.show()

#for test example 2, we define the objective function in BV excercise 9.30, which is the same as Hw 5
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
        gradient = np.full((5, 1), np.nan)
    else:
        fun = lambda A, x: -np.sum([np.log(1 - np.dot(A[:, i], x)) for i in range(A.shape[1])]) - np.sum(np.log(1 - x[i]**2) for i in range(len(x)))
        value= fun(A,x)
    #then we compute the gradient
        grad_fun = grad(fun, 1)
        gradient = grad_fun(A, x)
    #print('the gradient is',gradient)
    return value,gradient


data_rows = data_str.strip().split("\n")
data = [row.split("\t") for row in data_rows]
A2 = np.array(data, dtype=float)

x0 = np.zeros((100, 1))
tol = 1e-6
maxit = 2000
fun = lambda x: objective_function(A2,x)
M,m = 300,2 #estimation is also from homework last time

f_all12, gnorm_all12 = grad_method_fixed1(fun, x0, tol, maxit,m,M)
f_all22, gnorm_all22 = grad_method_fixed2(fun,x0,tol,maxit,m,M)
f_all32, gnorm_all32,_ = opt_grad_method(fun,x0,tol,maxit,m,M)

p_star12=f_all12[-1][0]
values_to_plot12 = f_all12 - p_star12
plt.semilogy(values_to_plot12.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with t=1/M')  # 图表标题
plt.show()

plt.semilogy(gnorm_all12, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for Gradient Method with t=1/M')  # 图表标题
plt.show()

p_star22=f_all22[-1][0]
values_to_plot22 = f_all22 - p_star22
plt.semilogy(values_to_plot22.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with t=2(M+m)')  # 图表标题
plt.show()

plt.semilogy(gnorm_all22, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for gradient method when t=2/(m+M)')  # 图表标题
plt.show()

p_star32=f_all32[-1][0]
values_to_plot32 = f_all32 - p_star32
plt.semilogy(values_to_plot32.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with optimal gradient method')  # 图表标题
plt.show()

plt.semilogy(gnorm_all32, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for optimal gradient')  # 图表标题
plt.show()

f_all, gnorm_all = gradmeth(fun, x0, tol, maxit)

p_star=f_all[-1][0]
values_to_plot = f_all - p_star
plt.semilogy(values_to_plot, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p*')  # 图表标题
plt.show()

hess = hessian(100,1,5)

x0 = np.zeros((10000, 1))
tol = 1e-6
maxit = 200
M,m = 100,1 #estimation is also from homework last time
fun = lambda x: worst_function(x,M,m)
f_all13, gnorm_all13 = grad_method_fixed1(fun, x0, tol, maxit,m,M)
f_all23, gnorm_all23 = grad_method_fixed2(fun,x0,tol,maxit,m,M)
f_all33, gnorm_all33,_ = opt_grad_method(fun,x0,tol,maxit,m,M)

p_star13=f_all13[-1][0]
values_to_plot13 = f_all13 - p_star13
plt.semilogy(values_to_plot13.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with t=1/M')  # 图表标题
plt.show()

plt.semilogy(gnorm_all13, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for Gradient Method with t=1/M')  # 图表标题
plt.show()

p_star23=f_all23[-1][0]
values_to_plot23 = f_all23 - p_star23
plt.semilogy(values_to_plot23.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with t=2(M+m)')  # 图表标题
plt.show()

plt.semilogy(gnorm_all23, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for gradient method when t=2/(m+M)')  # 图表标题
plt.show()

p_star33=f_all33[-1][0]
values_to_plot33 = f_all33 - p_star33
plt.semilogy(values_to_plot33.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with optimal gradient method')  # 图表标题
plt.show()

plt.semilogy(gnorm_all33, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for optimal gradient')  # 图表标题
plt.show()

#then we adjust M to 10000 and run the worst case example again
x0 = np.zeros((10000, 1))
tol = 1e-6
maxit = 200
M,m = 10000,1 #estimation is also from homework last time
fun = lambda x: worst_function(x,M,m)
f_all14, gnorm_all14 = grad_method_fixed1(fun, x0, tol, maxit,m,M)
f_all24, gnorm_all24 = grad_method_fixed2(fun,x0,tol,maxit,m,M)
f_all34, gnorm_all34,_ = opt_grad_method(fun,x0,tol,maxit,m,M)

p_star14=f_all14[-1][0]
values_to_plot14 = f_all14 - p_star14
plt.semilogy(values_to_plot14.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with t=1/M')  # 图表标题
plt.show()

plt.semilogy(gnorm_all14, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for Gradient Method with t=1/M')  # 图表标题
plt.show()

p_star24=f_all24[-1][0]
values_to_plot24 = f_all24 - p_star24
plt.semilogy(values_to_plot24.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with t=2(M+m)')  # 图表标题
plt.show()

plt.semilogy(gnorm_all24, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for gradient method when t=2/(m+M)')  # 图表标题
plt.show()

p_star34=f_all34[-1][0]
values_to_plot34 = f_all34 - p_star34
plt.semilogy(values_to_plot34.squeeze(), 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p* for Gradient Method with optimal gradient method')  # 图表标题
plt.show()

plt.semilogy(gnorm_all34, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('gradient norm')  # y轴标签
plt.title('Log Plot of gradient norm for optimal gradient')  # 图表标题
plt.show()

M,m = 100,1
n=10000
sparse_identity = eye(n, format='csr')
matrix1 =  hessian(M,m,n) - m*sparse_identity
matrix2 =  hessian(M,m,n) -M*sparse_identity

eigenvalues1, eigenvectors1 = eigs(matrix1, k=1, which='SM')
min_eigen = min(eigenvalues1)
print(min_eigen)
eigenvalues2, eigenvectors2 = eigs(matrix2, k=1, which='SM')
max_eigen = max(eigenvalues2)
print(max_eigen)

M,m = 10000,1
n=10000
sparse_identity = eye(n, format='csr')
matrix1 =  hessian(M,m,n) - m*sparse_identity
matrix2 =  hessian(M,m,n) -M*sparse_identity

eigenvalues1, eigenvectors1 = eigs(matrix1, k=1, which='SM')
min_eigen = min(eigenvalues1)
print(min_eigen)
eigenvalues2, eigenvectors2 = eigs(matrix2, k=1, which='SM')
max_eigen = max(eigenvalues2)
print(max_eigen)
