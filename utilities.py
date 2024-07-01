import sklearn
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import copy


def compute_surrogate(D):
    gpr = sklearn.gaussian_process.GaussianProcessRegressor(
        kernel=1.0 ** 2 * RBF(1.0, length_scale_bounds=(1e-3, 1e5)),
        alpha=0.001, n_restarts_optimizer=20, random_state=10)
    gpr.fit(D['x'], D['y'])
    return gpr


def normalize_D(D, center, length):
    Dnorm = copy.deepcopy(D)
    for i in range(D['x'].shape[1]):
        Dnorm['x'][:, i] = 2*(D['x'][:, i] - center[i]) / length[i]
    return Dnorm


def denormalize_x(xnorm, center, length):
    x = np.copy(xnorm)
    for i in range(xnorm.shape[1]):
        x[0, i] = (xnorm[0, i]*length[i]) / 2 + center[i]
    return x


def normalize_x(x, center, length):
    xnorm = np.copy(x)
    for i in range(x.shape[1]):
        xnorm[0, i] = 2*(x[0, i]-center[i]) / length[i]
    return xnorm


def f_local(x, problem, local_rank=0, nproc=4, A=None, b=None):
    # Given points x and problem name, compute outputs y
    y = np.zeros((x.shape[0], 1))
    if problem == 'hartman3':
        if nproc != 4:
            print('ERROR: hartman3 requires 4 agents\n')
        Ah = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        c = np.array([1, 1.2, 3, 3.2])
        p = np.array([[0.3689, 0.1170, 0.2673], [0.4699, 0.4837, 0.7470],
                      [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
        for i in range(x.shape[0]):
            temp = 0
            for j in range(3):
                temp -= ((x[i, j]-p[local_rank, j])**2)*Ah[local_rank, j]
            y[i, 0] = (-1)*c[local_rank]*np.exp(temp)
    elif problem == 'rosenbrock':
        for i in range(x.shape[0]):
            y[i, 0] = 100*(x[i, local_rank+1]-(x[i, local_rank])**2)**2\
                + (x[i, local_rank]-1)**2
    elif problem == 'camel':
        if nproc != 3:
            print('ERROR: camel requires 3 agents\n')
        for i in range(x.shape[0]):
            if local_rank == 0:
                y[i, 0] = (4-2.1*(x[i, 0]**2)+(x[i, 0]**4)/3)*(x[i, 0]**2)
            elif local_rank == 1:
                y[i, 0] = x[i, 0]*x[i, 1]
            else:
                y[i, 0] = (4*(x[i, 1]**2)-4)*(x[i, 1]**2)
    elif problem == 'beale':
        if nproc != 3:
            print('ERROR: beale requires 3 agents\n')
        for i in range(x.shape[0]):
            if local_rank == 0:
                y[i, 0] = (1.5-x[i, 0]+x[i, 0]*x[i, 1])**2
            elif local_rank == 1:
                y[i, 0] = (2.25-x[i, 0]+x[i, 0]*(x[i, 1]**2))**2
            else:
                y[i, 0] = (2.625-x[i, 0]+x[i, 0]*(x[i, 1]**3))**2
    elif problem == 'brent':
        if nproc != 3:
            print('ERROR: brent requires 3 agents\n')
        for i in range(x.shape[0]):
            if local_rank == 0:
                y[i, 0] = (x[i, 0]+10)**2
            elif local_rank == 1:
                y[i, 0] = (x[i, 1]+10)**2
            else:
                y[i, 0] = np.exp(-(x[i, 0]**2)-(x[i, 1]**2))
    elif problem == 'brown':
        if nproc != 3:
            print('ERROR: brown requires 3 agents\n')
        for i in range(x.shape[0]):
            y[i, 0] = (x[i, local_rank]**2)**(x[i, local_rank+1]**2 + 1)
            + (x[i, local_rank+1]**2)**(x[i, local_rank]**2 + 1)
    else:
        for i in range(x.shape[0]):
            y[i, 0] = np.linalg.norm(np.dot(A, x[i:(i+1), :].T)-b)**2
    return y
