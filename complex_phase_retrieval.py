import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#import random

def random_complex_vector(length=1, distribution='gaussian', param=1/np.sqrt(2)):
    """
    Returns a complex vector of dimension length x 1
    If distribution=='gaussian', complex elements are as x+iy, with x,y ~N(0,param^2), param is std
    If distribution=='uniform', complex elements have norm ]0,param] uniformly randomly, phase ]0,2pi] uniformly randomly
    If distribution=='fixed_norm', complex elements have norm param, phase ]0,2pi] uniformly randomly
    If distribution=='real_gaussian', x ~N(0,param^2), param is std, y=0
    If distribution=='real_uniform', x has norm ]0,param] uniformly randomly, y=0

    Complex standard normal (gaussian) distribution has variance 1/2 over the real and over the imaginary part (total variance 1)
    The real one has 1
    """
    assert distribution in ['gaussian','uniform','fixed_norm','real_gaussian','real_uniform'], \
        f"Parameter distribution can not be {distribution}"

    if distribution=='gaussian':
        return np.random.normal(loc=0,scale=param,size=length) + \
            (0+1j)*np.random.normal(loc=0,scale=param,size=length)
    elif distribution=='fixed_norm':
        return param*np.exp(2*np.pi*(0+1j)*np.random.random(length))
    elif distribution=='uniform':
        return param*(1 - np.random.random(length))*np.exp(2*np.pi*(0+1j)*np.random.random(length)) #to have norm in ]0,param]
    elif distribution=='real_gaussian':
        return param*np.random.normal(loc=0,scale=param,size=length)
    elif distribution=='real_uniform':
        return param*(1 - np.random.random(length))*np.random.choice([-1,1],1)

def define_w_hat(dim,complex=True):
    """
    Returns a complex vector of dimension d x 1, the "teacher" vector to be found.
    Its components are randomly initialized: its norm is in [0,1[, its phase in [0,2pi[.
    Its complex norm squared is d (which means its numpy.linalg.norm is the root of d)

    If complex=False, returns a real vector of squared norm d
    """
    #assert isinstance(dim,int) and dim > 0, f"Given variable dim should not be {dim}"
    if complex:
        ret = random_complex_vector(dim,'uniform',1)
    else:
        ret = random_complex_vector(dim,'real_uniform',1)
    return np.sqrt(dim)*ret/np.linalg.norm(ret)

def define_X(n,d,law='gaussian',param=1/np.sqrt(2)):
    """
    Returns a matrix of dimension n x d, that is the data.
    If law=='gaussian', its rows are complex standard normally distributed, meaning x+iy with x,y~N(0,1/2)
    If law=='real_gaussian', its rows are standard normally distributed, meaning x+iy with x~N(0,1), y=0
    """
    mat = np.empty((n,d),dtype=np.complex_)
    for i in range(n):
        mat[i] = random_complex_vector(d,law,param)
    return mat

def define_y(X,w_hat):
    """
    Returns an array of dimension n x 1, built from the data X and the teacher vector w_hat
    It keeps only the modulus of every element, that should be in principle in [0,1]
    (but can be bigger if the dimension of w_hat is finite)
    """
    return np.abs(X@w_hat)/np.sqrt(len(w_hat))

def cost(h,h_0):
    """
    The cost function "mu", comparing |X^i@w| to the value y^i=|X^i@w_hat| it corresponds to, minimized in that value
    USELESS wait this function is useless
    """
    return (np.abs(h)**2-np.abs(h_0)**2)**2/2

def cost_der_1(h,h_0):
    """
    The derivative of the cost function "mu" in its first argument, h
    """
    return 2*(np.abs(h)**2-np.abs(h_0)**2)*h

def isinbatch(b,n):
    """
    Gives a vector s_new with each element 1 with a probability b, or 0 otherwise.
    Vector of functions s^i(t)
    """
    s_new = np.empty(n)
    for i in range(n):
        s_new[i] = np.random.choice([1,0],p=[b,1-b])
    return s_new

def iterative_isinbatch(b,eta,tau,s_last):
    """
    The vector of functions s^i(t) when defined iteratively, takes its precedent state into account
    """
    s_new = np.empty(len(s_last))
    for i in range(len(s_last)):
        prob_1 = eta/tau*(1-s_last[i]) + (1-(1/b-1)*eta/tau)*s_last[i]
        prob_0 = 1-prob_1
        #print(f'run {i} and s_last {s_last[i]}: Probabilities are {prob_1} for 1 and {prob_0} for 0')
        s_new[i] = np.random.choice([1,0],p=[prob_1,prob_0])
    return s_new

def loss(w,X,y,s_last,b):
    """
    The loss function to be minimized, y^i can be taken in place of \hat{h}^.
    
    """
    return np.sum(s_last/len(y)/b*cost(X@w/np.sqrt(len(w)),y))

def loss_grad(w,X,y,s_last,b):
    """
    The gradient in w of the loss function
    """
    ret = np.empty(len(w),dtype=np.complex_)
    for k in range(len(w)):
        ret[k] = s_last.T@(cost_der_1(X@w/np.sqrt(len(w)),y)*X[:,k].conj())/np.sqrt(len(w))/len(y)/b
    return ret

def magnetization_norm(w,w_hat):
    return np.abs(w.conj().T@w_hat/len(w_hat))

def define_w_0(m_0, w_hat, complex=True):
    """
    The initialization of vector w, "warm initialization" to avoid getting stuck in a perpendicular
    state to w_hat
    """
    if complex:
        z = random_complex_vector(len(w_hat))
    else:
        z = random_complex_vector(len(w_hat),'real_gaussian',1)
    coeff = np.sqrt(len(w_hat)*(1-m_0**2))/np.linalg.norm(z)
    vec = m_0*w_hat+coeff*z
    return vec/np.linalg.norm(vec)*np.sqrt(len(w_hat))

def w_next(w,X,y,b,eta,tau,s_last):
    """
    The recursive algorithm that links all together
    """
    cal = w - eta*loss_grad(w,X,y,s_last,b)
    
    return cal/np.linalg.norm(cal)*np.sqrt(len(w))

def initialize(N, d, eta, tau, b, m_0, iter_max, isComplex):

    iter_max = int(iter_max)

    #eta must be smaller than tau
    #b must be bigger than eta/(tau+eta)

    assert eta <= tau, "eta must be bigger than tau"
    assert (b >= eta/(tau+eta)), "b must be bigger than eta/(tau+eta)"

    if isComplex:
        X = define_X(N,d,'gaussian')
    else:
        X = define_X(N,d,'real_gaussian',1)
    
    w_hat = define_w_hat(d,complex=isComplex)
    y = define_y(X,w_hat)

    m_norm_all = np.empty(iter_max)
    loss_all = np.empty(iter_max)
    s_vector = np.empty(N)
    w = define_w_0(m_0,w_hat,complex=isComplex)

    return X, w_hat, y, m_norm_all, loss_all, s_vector, w

def loop(N=100, d=30, eta=1, tau=10, b=0.1, m_0=0.2, iter_max=1e3, isComplex=True, np_rd_seed=None):

    np.random.seed(np_rd_seed)
    #random.seed(0) if actively using random

    X, w_hat, y, m_norm_all, loss_all, s_vector, w = initialize(N, d, eta, tau, b, m_0, iter_max, isComplex)

    iter_max = int(iter_max)
    
    s_vector = isinbatch(b,N) #to "initialize" s, actually havine s for t=0
    for iter in range(iter_max): #iteration is t

        w = w_next(w,X,y,b,eta,tau,s_vector)
        m_norm_all[iter] = magnetization_norm(w,w_hat)
        loss_all[iter] = loss(w,X,y,s_vector,b)
        s_vector = iterative_isinbatch(b,eta,tau,s_vector) #"useless" on the last run but this way allows the prior isinbatch to be called for t=0

    return m_norm_all, loss_all

def plot_magLoss_iter(m_norm_all,loss_all,iter_max):

    plt.subplot(1,2,1)
    plt.plot(np.arange(0,iter_max,1),m_norm_all)
    plt.xlabel('t')
    plt.ylabel('|m|(t)')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0,iter_max,1),loss_all)
    plt.xlabel('t')
    plt.ylabel('$\mathcal{L}(t)$')
    plt.show()

def plot_descent_methods(m_norm, loss, labels, iter_max): #the m_norm and loss must be narrays of dim (diff_graphs, values)

    plt.subplot(1,2,1)
    plt.plot(np.arange(0,iter_max,1).T,m_norm.T,)
    plt.xlabel('t')
    plt.ylabel('|m|(t)')
    plt.xscale('log')
    plt.legend(labels)
    plt.subplot(1,2,2)
    plt.plot(np.arange(0,iter_max,1).T,loss.T)
    plt.xlabel('t')
    plt.ylabel('$\mathcal{L}(t)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(labels)
    plt.show()

def main_simple():

    N = 100
    d = 30
    eta = 1 # eta must be smaller than tau
    b = 0.1
    tau = 10 # b must be bigger than eta/(tau+eta)
    m_0 = 0.2
    iter_max = 1e3
    isComplex = True
    np_rd_seed = 0 # for the results to be reproductible


    m_norm_all, loss_all = loop(N, d, eta, tau, b, m_0, iter_max, isComplex, np_rd_seed)

    plot_magLoss_iter(m_norm_all, loss_all, iter_max)

def main_comparaison_methods():
    N = 30
    d = 10
    eta = 0.01 # eta must be smaller than tau
    b = np.array([1., 0.5, 0.5])
    tau = np.array([1., eta/0.5, 1.]) # b must be bigger than eta/(tau+eta)
    m_0 = 0.2
    iter_max = 1e3
    isComplex = False
    np_rd_seed = np.arange(0,5,1) # for the results to be reproductible

    graph_labels = ['GD','SGD','p-SGD']

    m_graph, loss_graph = np.empty((3,int(iter_max))), np.empty((3,int(iter_max)))

    for descent_type in range(3): # for each descent type, 500 different loops are taken over the narray np_rd_seed
        m_to_average, loss_to_average = np.empty((len(np_rd_seed),int(iter_max))), np.empty((len(np_rd_seed),int(iter_max)))
        for sample in tqdm(range(len(np_rd_seed))):
            m_to_average[sample], loss_to_average[sample] = loop(N, d, eta, tau[descent_type], b[descent_type], m_0, iter_max, isComplex, np_rd_seed[sample])
        m_graph[descent_type], loss_graph[descent_type] = np.mean(m_to_average,axis=0), np.mean(loss_to_average,axis=0)

    data_graph = np.concatenate((m_graph,loss_graph))

    np.savetxt("methods_comparaison.csv", data_graph, fmt="%.6f")

    #data_graph = np.genfromtxt('methods_comparaison.csv')
    #m_graph = data_graph[0:3]
    #loss_graph = data_graph[3:6]

    plot_descent_methods(m_graph, loss_graph, graph_labels, int(iter_max))

if __name__ == "__main__":
    main_comparaison_methods()