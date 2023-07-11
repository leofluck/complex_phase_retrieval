import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import pickle
try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x
#import random


name = 'alpha_2_75'


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
    #assert distribution in ['gaussian','uniform','fixed_norm','real_gaussian','real_uniform'], \
    #    f"Parameter distribution can not be {distribution}"

    if distribution=='gaussian':
        return np.random.normal(loc=0,scale=param,size=length) + (0+1j)*np.random.normal(loc=0,scale=param,size=length)
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
    return random_complex_vector(np.array([n,d]),law,param)

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
    return (np.abs(h)**2-np.abs(h_0)**2)*h.conj()

def isinbatch(b,n):
    """
    Gives a vector s_new with each element 1 with a probability b, or 0 otherwise.
    Vector of functions s^i(t)
    """
    s_new = np.random.choice(a=[1,0],p=[b,1-b],size=n)
    return s_new

def iterative_isinbatch(b,eta,tau,s_last):
    """
    The vector of functions s^i(t) when defined iteratively, takes its precedent state into account
    """

    prob_1 = eta/tau*(1-s_last) + (1-(1/b-1)*eta/tau)*s_last

    vec = np.random.random(len(s_last))
    
    return np.where(prob_1 > vec,1,0)


def loss(w,X,y,s_last,b):
    """
    The loss function to be minimized, y^i can be taken in place of \hat{h}^.
    
    """
    return np.sum(s_last/b*cost(X@w/np.sqrt(len(w)),y))

def loss_grad(w,X,y,s_last,b):
    """
    The gradient in w of the loss function
    """

    ret = s_last.T@np.multiply(X.T,cost_der_1(X@w/np.sqrt(len(w)),y)).T/np.sqrt(len(w))/b
    return (ret.T).conj()

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
    coeff = (-m_0*np.real(w_hat.T@np.conj(z)) - np.sqrt((m_0*np.real(w_hat.T@np.conj(z)))**2-z.T@np.conj(z)*(m_0**2*w_hat.T@w_hat.conj()-len(w_hat))))/np.linalg.norm(z)**2
    vec = m_0*w_hat+coeff*z
    return vec/np.linalg.norm(vec)*np.sqrt(len(w_hat))

def w_next(w,X,y,b,eta,tau,s_last):
    """
    The recursive algorithm that links all together
    """
    cal = w - eta*loss_grad(w,X,y,s_last,b)
    
    return cal/np.linalg.norm(cal)*np.sqrt(len(w))

def initialize(N, d, eta, tau, b, m_0, iter_max, isComplex, sampling): #sampling assumed logarithmically

    iter_max = int(iter_max)

    #eta must be smaller than tau
    #b must be bigger than eta/(tau+eta)

    assert eta <= tau, "eta must be smaller than tau"
    assert (b >= eta/(tau+eta)), "b must be bigger than eta/(tau+eta)"

    if isComplex:
        X = define_X(N,d,'gaussian')
    else:
        X = define_X(N,d,'real_gaussian',1)
    
    w_hat = define_w_hat(d,complex=isComplex)
    y = define_y(X,w_hat)

    m_norm_all = np.empty(len(sampling))
    loss_all = np.empty(len(sampling))
    s_vector = np.empty(N)
    w = define_w_0(m_0,w_hat,complex=isComplex)

    #print(magnetization_norm(w,w_hat))

    return X, w_hat, y, m_norm_all, loss_all, s_vector, w

def loop(N=100, d=30, eta=1, tau=10, b=0.1, m_0=0.2, iter_max=1e3, isComplex=True, np_rd_seed=None, sampling=np.arange(999)): #sampling assumed logarithmially
    
    np.random.seed(np_rd_seed)
    #random.seed(0) if actively using random

    X, w_hat, y, m_norm_all, loss_all, s_vector, w = initialize(N, d, eta, tau, b, m_0, iter_max, isComplex, sampling)

    iter_max = int(iter_max)
    
    s_vector = isinbatch(b,N) #to "initialize" s, actually havine s for t=0
    for iter in tqdm(range(iter_max)): #iteration is t
        if iter in sampling:
            idx = np.where(sampling == iter)[0][0]
            m_norm_all[idx] = magnetization_norm(w,w_hat)
            loss_all[idx] = loss(w,X,y,s_vector,b)
        w = w_next(w,X,y,b,eta,tau,s_vector) #'useless' on the last run but allows t=0 to be on graph
        s_vector = iterative_isinbatch(b,eta,tau,s_vector) #"useless" on the last run but this way allows the prior isinbatch to be called for t=0

    return m_norm_all, loss_all


def plot_descent_methods(sampling, m_norm, loss, labels): #the m_norm and loss must be narrays of dim (diff_graphs, values)

    plt.subplot(1,2,1)
    plt.plot(sampling,m_norm.T)
    plt.xlabel('t/$\eta$')
    plt.ylabel('|m|(t)')
    plt.xscale('log')
    plt.legend(labels)
    plt.subplot(1,2,2)
    plt.plot(sampling,loss.T)
    plt.xlabel('t/$\eta$')
    plt.ylabel('$\mathcal{L}(t)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(labels)
    plt.savefig(f'{name}.png')


def main_final():
    N = 1375
    d = 500
    eta = 0.05 # eta must be smaller than tau
    b = np.array([1., 0.5, 0.5])
    tau = np.array([1., eta/0.5, 1.]) # b must be bigger than eta/(tau+eta)
    m_0 = 0.2
    iter_max = 1e6
    isComplex = True

    nb_samples = 1000

    assert (nb_samples <= iter_max), "nb samples should be smaller than nb of timesteps"

    #np_rd_seed = np.arange(0,1,1) # for the results to be reproductible, the length of this object is the number of runs which get averaged
    #np_rd_seed = np.random.randint(0,1000,3)
    try:
        np.random.seed(int(argv[1]))
    except:
        emptyvar=None
    #nb_runs = 1

    graph_labels = ['GD','SGD','p-SGD']

    sampling = np.unique(np.round(np.logspace(0,np.log(iter_max-1)/np.log(10),nb_samples)).astype(int))

    m_graph, loss_graph = np.empty((3,len(sampling))), np.empty((3,len(sampling)))

    for descent_type in range(3): # for each descent type, 500 different loops are taken over the narray np_rd_seed
        #m_to_average, loss_to_average = np.empty((nb_runs,int(iter_max))), np.empty((nb_runs,int(iter_max)))
        #for sample in range(nb_runs):
            #m_to_average[sample], loss_to_average[sample] = loop(N, d, eta, tau[descent_type], b[descent_type], m_0, iter_max, isComplex)
        #print(f'descent type: {descent_type}')
        m_graph[descent_type], loss_graph[descent_type] = loop(N, d, eta, tau[descent_type], b[descent_type], m_0, iter_max, isComplex, None, sampling)

    data_graph = np.concatenate(([sampling],m_graph,loss_graph))

    #print('saving soon')

    try:
        data_file = open(f'{name}/data_run_{argv[1]}.pickle','wb')
        pickle.dump(data_graph,data_file)
        data_file.close()
    except:
        data_file = open(f'{name}/data_run_0.pickle','wb')
        pickle.dump(data_graph,data_file)
        data_file.close()

def main_concatenate(runs,raw,saving): #if raw, reads all the parallel data. Else, reads the already concatenated data and plots. If saving==True, saves the concatenated data in a pickle. Graphs anyways
    
    graph_labels = ['GD','SGD','p-SGD']

    if raw:
        data_file = open(f'{name}/data_run_0.pickle','rb') #assuming there always is a file
        data = pickle.load(data_file)
        samples = data[0]
        data_file.close()

        mag_all = np.empty((3,runs,len(samples)))
        loss_all = np.empty((3,runs,len(samples)))

        for i in range(runs):
            data_file = open(f'{name}/data_run_{i}.pickle','rb')
            data = pickle.load(data_file)
            mag_all[:,i,:] = data[1:4]
            loss_all[:,i,:] = data[4:7]
            data_file.close()

        if saving:
            data_file = open(f'results_{name}.pickle','wb')
            data = np.empty((7,runs,len(samples)))
            data[0] = np.tile(samples,(runs,1))
            data[1:4] = mag_all
            data[4:7] = loss_all
            pickle.dump(data,data_file)
            data_file.close()
        
        plot_descent_methods(samples, mag_all.mean(axis=1), loss_all.mean(axis=1), graph_labels)

    else:
        data_file = open(f'results_{name}.pickle','rb')
        data = pickle.load(data_file)
        plot_descent_methods(data[0][0], data[1:4].mean(axis=1), data[4:7].mean(axis=1), graph_labels)
        data_file.close()


if __name__ == "__main__":

    run = True #run ou plot

    if run:
        main_final()
    else:
        main_concatenate(50,raw=True,saving=True)