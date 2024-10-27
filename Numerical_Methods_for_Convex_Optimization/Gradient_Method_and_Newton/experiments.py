import autograd.numpy as np  # 使用autograd的numpy版本
from autograd import grad
from sympy import symbols, solve, lambdify
import copy
from algorithms import gradmeth, objective_function, newtmeth, objective_function2
from data import data_str
import matplotlib.pyplot as plt

# The test case
def quad(x, A, b):
    f = 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x)
    g = np.dot(A, x) + b
    return f, g

n = 5
A = np.linalg.inv(np.array([[1/(i+j-1) for i in range(1, n+1)] for j in range(1, n+1)]))
b = np.ones((n, 1))
fun = lambda x: quad(x, A, b)
x0 = np.ones((n, 1))
tol = 1e-6
maxit = 100

f_all, gnorm_all = gradmeth(fun, x0, tol, maxit)

print(f"All function values: {f_all}")
print(f"All gradient norms: {gnorm_all}")

#(a), compute the minimizer -A^(-1)b
A_inv = np.linalg.inv(A)
x = -np.dot(A_inv, b)
print('the minimizer x is',x)
#compute the optimal value
p_star = fun(x)[0][0][0]
print('the optimal value is',p_star)
#compute the value at initial point
f_0 = fun(x0) [0][0][0]
#compute the difference at x0
difference_0 = f_0 - p_star
print('this is the difference at the original time step',difference_0)
#compute the difference after 100 iterations
difference_100 = f_all[-1][0] - p_star
print('this is the difference after 100 timesteps', difference_100)
#computer the proportion of difference 100 wrt difference 0
proportion = difference_100/difference_0
print('this is the factor that the difference is reduced after 100 steps',proportion)

#calculate eigenvalues of hilbert matrix A to obtain M and m
eigenvalues, _ = np.linalg.eig(A)
print('the eigenvalues are',eigenvalues)

#so we get that the minimum eigenvalue is 6.38141450e-01, the maximum eigenvalue is 3.04142842e+05
alpha=0.25
beta=0.5
c = 1-alpha*beta*2*(6.38141450e-01/3.04142842e+05)
print('c is',c)
print('the theoratical factor is',c**100)

data_rows = data_str.strip().split("\n")
data = [row.split("\t") for row in data_rows]
A2 = np.array(data, dtype=float)
print(A2)
print(A2.shape)

# run the gradient method on this new function
x0 = np.zeros((100, 1))
tol = 1e-6
maxit = 100
fun = lambda x: objective_function(A2,x)
f_all, gnorm_all = gradmeth(fun, x0, tol, maxit)
print(f_all[-1])

p_star=f_all[-1][0]
values_to_plot = f_all - p_star
plt.semilogy(values_to_plot, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of f(x(k)) - p*')  # 图表标题
plt.show()

#then plot the gradient norms
plt.semilogy(gnorm_all, 'x')  # 'x'指定了使用x标记每个点
plt.xlabel('Iteration k')  # x轴标签
plt.ylabel('f(x(k)) - p*')  # y轴标签
plt.title('Log Plot of gradient norm')  # 图表标题
plt.show()

#estimate the condition number M/m in this case, we first estimate c

value = values_to_plot[-1]/values_to_plot[0]
eq = c**70 - value

print(values_to_plot[-1], values_to_plot[0])

solutions = solve(eq, c)

if solutions:
    print(f"When c^100 = {value}, c = {solutions[0]}")
else:
    print("No explicit solution")

def equation(c):
    return c**70 - value


c_num = lambdify(c, solutions[0], 'numpy')
print(f"Numerical solution: c = {c_num()} ")

#c=1-2 beta*alpha * 1/condition_number

condition_number = 2*alpha*beta/(c_num-1)
print('the estimated condition number is',condition_number)

def quad(x, A, b):
    f = 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x)
    g = np.dot(A, x) + b
    hessian_f = A
    return f, g, hessian_f


n = 5
A = np.linalg.inv(np.array([[1/(i+j-1) for i in range(1, n+1)] for j in range(1, n+1)]))
b = np.ones((n, 1))
fun = lambda x: quad(x, A, b)
x0 = np.ones((n, 1))
tol = 1e-6
maxit = 100

x_lst2, f_all2, gradient_norm2,  hessian2 = newtmeth(fun, x0, tol, maxit)

print(f"All function values: {f_all2}")
print(f"All hessian: {hessian2}")

x0 = np.zeros((100, 1))
tol = 1e-8
maxit = 100
fun = lambda x: objective_function2(A2,x)
x3, f_all3, gnorm_all3, hessian3 = newtmeth(fun, x0, tol, maxit)
print(f_all3[-1])

plt.figure()
p_star3 = f_all3[-1][0]
values_to_plot3 = f_all3 - p_star3
plt.semilogy(values_to_plot, 'x', color = 'blue',  label='Grad Method')
plt.semilogy(values_to_plot3,'x' ,color='red', label='Newt Method')
plt.title('Log Plot of f(x(k)) - p*')  # 图表标题
plt.show()

plt.semilogy(gnorm_all, 'x',color = 'blue',  label='Grad Method')
plt.semilogy(gnorm_all3, 'x',color='red', label='Newt Method')
plt.title('Log Plot of grad norm')
plt.show()

#first we get the largest and smallest eigenvalue of hessian matrix, M and m.

#get several Hessian matrices from the hessian 3 list (infact there're only six of them so we can get them all)

eigen1, _ = np.linalg.eig(hessian3[0])
eigen2, _ = np.linalg.eig(hessian3[1])
eigen3, _ = np.linalg.eig(hessian3[2])
eigen4, _ = np.linalg.eig(hessian3[3])
eigen5, _ = np.linalg.eig(hessian3[4])
eigen6, _ = np.linalg.eig(hessian3[5])

eigen_list = [eigen1,eigen2,eigen3,eigen4,eigen5,eigen6]

M = eigen1[0]
m = eigen1[0]

for i in range(len(eigen_list)):
    for j in range(len(eigen_list[i])):
        if eigen_list[i][j]> M:
            M = eigen_list[i][j]
        elif eigen_list[i][j]< m:
            m = eigen_list[i][j]

print('the largest eigen value M is', M)
print('the smallest eigen value m is', m)

#then we use similar method to estimate L
hess_diff_lst = []
for i in range(len(hessian3)-1):
    hess_diff_lst.append(hessian3[i+1]-hessian3[i])

print(len(x3))
x_diff_lst = []
for i in range(len(x3)-1):
    x_diff_lst.append(x3[i+1]-x3[i])

#print(x_diff_lst)
l2_hess_lst = [np.linalg.norm(hess,'fro') for hess in hess_diff_lst]
l2_x_lst = [np.linalg.norm(x) for x in x_diff_lst]

#print(len(l2_hess_lst),len(l2_x_lst))
print(l2_hess_lst)
print(l2_x_lst)
quotient_lst = [l2_hess_lst[i]/l2_x_lst[i] for i in range(len(l2_hess_lst))]

L=max(quotient_lst)
print('the estimated L is',L)

#then we use similar method to estimate L
hess_diff_lst = []
for i in range(len(hessian3)-1):
    hess_diff_lst.append(hessian3[i+1]-hessian3[i])

print(len(x3))
x_diff_lst = []
for i in range(len(x3)-1):
    x_diff_lst.append(x3[i+1]-x3[i])

#print(x_diff_lst)
l2_hess_lst = [np.linalg.norm(hess,'fro') for hess in hess_diff_lst]
l2_x_lst = [np.linalg.norm(x) for x in x_diff_lst]

#print(len(l2_hess_lst),len(l2_x_lst))
print(l2_hess_lst)
print(l2_x_lst)
quotient_lst = [l2_hess_lst[i]/l2_x_lst[i] for i in range(len(l2_hess_lst))]

L=max(quotient_lst)
print('the estimated L is',L)

 #then we compute the upper bound of the number of steps by computing the steps needed under the linear convergent rate
#this happens when the distance to the optimal point is less than eta
upper_steps = values_to_plot[0]/gamma
print('the upper bound of steps is ',upper_steps)
#then we compute the lower bound of steps by computing the steps neede under quatratic convergence rate
#this happen when the distance to optimal point is greater than eta
#print(np.log2(np.log2((2*m**3/L**2)/10**(-8))))
lower_steps = np.log2(np.log2((2*m**3/L**2)/10**(-8)))
print('the lower bound of steps is',lower_steps)

