import autograd.numpy as np
from autograd import grad


def mxv(v, f, x):
    eta = 1e-6
    x_succ = x + eta*v
    return (grad(f)(x_succ) - grad(f)(x))/eta

def gradfi(d, f, x):
    return mxv(d,f,x) + grad(f)(x)



def armijo(f, x_k, d):
    alpha = 1
    gamma = 1e-3

    while True:
        if f(x_k + alpha*d) > f(x_k) + alpha*gamma*(grad(f)(x_k) @ d):
            alpha = 0.5*alpha
        else:
            return alpha
        
        
        
def dt(f, x, k):

    epsilon_1 = 0.5
    epsilon_2 = 0.5
    p = 0

#     s = -(gradfi(p, f, x))
    s = -grad(f)(x)
    
    if (s @ mxv(s, f, x)) < (epsilon_1 * (np.linalg.norm(s))**2):
        d = -(grad(f)(x))
        return d
    
    while True:
        
        if (s @ mxv(s, f, x)) <= 1e-6:              #default = 1e-9
            return -grad(f)(x)
        
        alfa = -((gradfi(p, f, x) @ s) / (s @ mxv(s, f, x)))
        p = p + alfa * s

        if np.linalg.norm(gradfi(p, f, x)) <= (1/(k+1))*epsilon_2*(np.linalg.norm(grad(f)(x))):
            d = p
            return d
        
        else:
            beta = (gradfi(p, f, x) @ mxv(s, f, x)) / (s @ mxv(s, f, x))
            s = -(gradfi(p, f, x)) + beta * s

            if (s @ mxv(s, f, x)) < (epsilon_1 * (np.linalg.norm(s))**2):
                d = p
                return d
            
            
def nt(f, x, eps = 1e-5):  
    
    k = 0
    
    if np.linalg.norm(grad(f)(x)) < eps:
        return x
    
    while True:

        d = dt(f, x, k)
        a = armijo(f, x, d)
        x = x + a * d
        
        if np.linalg.norm(grad(f)(x)) <= eps:
            return x

        k = k + 1