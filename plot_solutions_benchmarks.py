import numpy as np
import dill as pickle
from utilities import f_local
import matplotlib
import matplotlib.pyplot as plt
from disropt.problems import Problem
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# %%
# Select the stepsize value related to the results you want to plot
stepsize = 0.01
# Select the problem name related to the results you want to plot
problem = 'camel'
# %%
# Load all the data for the plots and initialize needed objects
with open('./Data/{}/agents_step_{}.npy'.format(problem, stepsize), 'rb') as f:
    nproc = np.load(f)
    n = np.load(f)
    iterations_scaled = np.load(f)
    N_test = np.load(f)
    nr_init = np.load(f)
iterations = iterations_scaled*nproc
xaxis = np.arange(1, iterations+1)
figure_counter = 1
x_agent = []
x_agent_exploit = []
local_function = []
dictionaries = []
scaling = []
if problem == 'ls':
    A = []
    b = []
fvalue = np.zeros((N_test, iterations))
cons_err = np.zeros((N_test, iterations))
surr_err = np.zeros((N_test, iterations))
fvalue_exploit = np.zeros((N_test, iterations))
cons_err_exploit = np.zeros((N_test, iterations))
surr_err_exploit = np.zeros((N_test, iterations))
for i in range(nproc):
    filename = './Data/{}/agent_{}_scaling_step_{}.npy'.format(problem, i,
                                                               stepsize)
    scaling.append(np.load(filename))
    with open('./Data/{}/agent_{}_function_step_{}.pkl'.format(problem, i,
                                                               stepsize),
              'rb') as inp:
        local_function.append(pickle.load(inp))
    filename = './Data/{}/agent_{}_sequence_step_{}.npy'.format(problem, i,
                                                                stepsize)
    x_agent.append(np.load(filename))
    filename = './Data/{}/agent_{}_exploitation_step_{}.npy'.format(problem, i,
                                                                    stepsize)
    x_agent_exploit.append(np.load(filename))
    filename = './Data/{}/agent_{}_dictionaries_step_{}.npy'.format(problem, i,
                                                                    stepsize)
    dictionaries.append(np.load(filename))
    if problem == 'ls':
        filename = './Data/{}/agent_{}_A_step_{}.npy'.format(problem, i,
                                                             stepsize)
        A.append(np.load(filename))
        filename = './Data/{}/agent_{}_b_step_{}.npy'.format(problem, i,
                                                             stepsize)
        b.append(np.load(filename))
with open('./Data/{}/constraints_step_{}.pkl'.format(problem, stepsize),
          'rb')as input:
    constr = pickle.load(input)
# %%
# Compute results
# '_exploit' objects refer to points computed without the IDWs
# Few parts of the code are repeated, to have the possibility of commenting out
# portions of the code not of interest
for idx_test in range(N_test):
    x_avg = np.zeros((n, iterations))
    x_avg_exploit = np.zeros((n, iterations))
    for i in range(nproc):
        x_avg += x_agent[i][idx_test]
        x_avg_exploit += x_agent_exploit[i][idx_test]
    x_avg /= nproc
    x_avg_exploit /= nproc
    # Compute function values in x_avg
    avg_values = np.zeros((iterations, 1))
    if problem == 'ls':
        AtA = np.zeros((n, n))
        ATb = np.zeros((n, 1))
        solution_value = 0
        opt_value = 0
        for i in range(nproc):
            AtA += np.dot(A[i].T, A[i])
            ATb += np.dot(A[i].T, b[i])
            avg_values += f_local(x_avg.T, problem, A=A[i], b=b[i])
        x_solution = np.linalg.solve(AtA, ATb)
        for i in range(nproc):
            opt_value += f_local(x_solution.T, problem, A=A[i], b=b[i])
    else:
        if problem == 'hartman3':
            x_solution = np.array([0.1140, 0.556, 0.852])
        elif problem == 'camel':
            x_solution = np.array([-0.0898, 0.7126])
        elif problem == 'beale':
            x_solution = np.array([3, 0.5])
        elif problem == 'brent':
            x_solution = np.array([-10, -10])
        elif problem == 'brown':
            x_solution = np.array([0, 0, 0, 0])
        else:
            x_solution = np.ones((n, 1))
        opt_value = 0
        for i in range(nproc):
            avg_values += f_local(x_avg.T, problem, i, nproc)
            opt_value += f_local(x_solution.reshape((1, n)), problem, i, nproc)
    fvalue[idx_test, :] = avg_values.reshape((1, iterations))
    # Uncomment code below to plot all the single realizations
    # plt.figure(figure_counter)
    # figure_counter += 1
    # plt.semilogy(xaxis, fvalue[idx_test, :],
    #              label='Proposed method - exploration')
    # plt.legend()
    # plt.xlabel('Iteration number')
    # plt.ylabel('Function value')
    # plt.xlim(1, iterations)
    # plt.hlines(opt_value, xaxis[0], xaxis[-1], color='red')
    # plt.show()
    ###
    # Compute function values in x_avg_exploit
    avg_values = np.zeros((iterations, 1))
    if problem == 'ls':
        AtA = np.zeros((n, n))
        ATb = np.zeros((n, 1))
        solution_value = 0
        opt_value = 0
        for i in range(nproc):
            AtA += np.dot(A[i].T, A[i])
            ATb += np.dot(A[i].T, b[i])
            avg_values += f_local(x_avg_exploit.T, problem, A=A[i], b=b[i])
        x_solution = np.linalg.solve(AtA, ATb)
        for i in range(nproc):
            opt_value += f_local(x_solution.T, problem, A=A[i],
                                 b=b[i])
    else:
        if problem == 'hartman3':
            x_solution = np.array([0.1140, 0.556, 0.852])
        elif problem == 'camel':
            x_solution = np.array([-0.0898, 0.7126])
        elif problem == 'beale':
            x_solution = np.array([3, 0.5])
        elif problem == 'brent':
            x_solution = np.array([-10, -10])
        elif problem == 'brown':
            x_solution = np.array([0, 0, 0, 0])
        else:
            x_solution = np.ones((n, 1))
        opt_value = 0
        for i in range(nproc):
            avg_values += f_local(x_avg_exploit.T, problem, i, nproc)
            opt_value += f_local(x_solution.reshape((1, n)), problem, i,
                                 nproc)
    fvalue_exploit[idx_test, :] = avg_values.reshape((1, iterations))
    # Uncomment code below to plot all the single realizations
    # plt.figure(figure_counter)
    # figure_counter += 1
    # plt.plot(xaxis, fvalue_exploit[idx_test, :],
    #          label='Proposed method - exploitation')
    # plt.legend()
    # plt.xlabel('Iteration number')
    # plt.ylabel('Function value')
    # plt.xlim(1, iterations)
    # plt.hlines(opt_value, xaxis[0], xaxis[-1], color='red')
    # plt.show()
    ###
    # Solve global surrogate problem (sum of the finallocal surrogates)
    # with a solver.
    # Compute relative error with respect to the solution
    # value for the average and for all the agents
    global_obj_func = 0
    for i in range(nproc):
        global_obj_func += local_function[i][idx_test]
    global_pb = Problem(global_obj_func, constr)
    x_surr_solution = global_pb.solve()
    cost_surr = global_obj_func.eval(x_surr_solution)
    avg_values_surr = []
    agents_loss = []
    for i in range(iterations):
        avg_values_surr.append(global_obj_func.eval(x_avg[:, i].reshape(n, 1)))
    for i in range(nproc):
        agents_loss_single = []
        for j in range(iterations):
            agents_loss_single.append(global_obj_func.eval(
                x_agent[i][idx_test][:, j].reshape(n, 1)).reshape(1))
        agents_loss.append(agents_loss_single)
    surr_err[idx_test, :] = (abs(np.array(avg_values_surr).reshape(iterations,
                                                                   1)
                                 - np.ones((iterations, 1))*cost_surr)
                             / abs(cost_surr)).T
    # Uncomment code below to plot all the single realizations
    # plt.figure(figure_counter)
    # figure_counter += 1
    # for i in range(nproc):
    #     plt.semilogy(xaxis, abs(agents_loss[i]-np.ones((iterations, 1))
    #                             * cost_surr)/abs(cost_surr),
    #                   label='agent {}'.format(i))
    # plt.semilogy(xaxis, surr_err[idx_test, :], label='average')
    # plt.legend()
    # plt.xlabel('Iteration number')
    # plt.ylabel('Distance from surrogate solution')
    # plt.ylim(1e-12, 100)
    # plt.show()
    ###
    # Compute distance of each agent from x_avg at the end of each experiment
    c_err = []
    for i in range(iterations):
        err = []
        for j in range(nproc):
            err.append(np.linalg.norm(x_avg[:, i]-x_agent[j][idx_test][:, i]))
        c_err.append(max(err))
    cons_err[idx_test, :] = c_err
    # Uncomment code below to plot all the single realizations
    # plt.figure(figure_counter)
    # figure_counter += 1
    # plt.semilogy(xaxis, c_err, label='Proposed method - exploration')
    # plt.legend()
    # plt.xlabel('Iteration number')
    # plt.ylabel('Consensus error')
    # plt.show()
    ###
    # Compute distance of each agent from x_avg_exploit
    # at the end of each experiment
    c_err = []
    for i in range(iterations):
        err = []
        for j in range(nproc):
            err.append(np.linalg.norm(x_avg_exploit[:, i]
                                      - x_agent_exploit[j][idx_test][:, i]))
        c_err.append(max(err))
    cons_err_exploit[idx_test, :] = c_err
    # Uncomment code below to plot all the single realizations
    # plt.figure(figure_counter)
    # figure_counter += 1
    # plt.semilogy(xaxis, c_err, label='Proposed method - exploitation')
    # plt.legend()
    # plt.xlabel('Iteration number')
    # plt.ylabel('Consensus error')
    # plt.show()
# %%
# Plot function values average of x_avg: median, minimum, and maximum
fvalue_median = np.median(fvalue, 0)
fvalue_max = fvalue[np.argmax(np.sum(fvalue, 1)), :]
fvalue_min = fvalue[np.argmin(np.sum(fvalue, 1)), :]
plt.figure(figure_counter)
figure_counter += 1
plt.plot(xaxis, fvalue_max, label='worst')
plt.plot(xaxis, fvalue_median, label='median')
plt.plot(xaxis, fvalue_min, label='best')
plt.legend()
plt.xlabel('Iteration number')
plt.ylabel('Function value')
plt.xlim(1, iterations)
plt.hlines(opt_value, xaxis[0], xaxis[-1], color='red')
plt.show()
# Plot function values average of x_avg_exploit: median, minimum, and maximum
fvalue_median = np.median(fvalue_exploit, 0)
fvalue_max = fvalue_exploit[np.argmax(np.sum(fvalue_exploit, 1)), :]
fvalue_min = fvalue_exploit[np.argmin(np.sum(fvalue_exploit, 1)), :]
plt.figure(figure_counter)
figure_counter += 1
plt.plot(xaxis, fvalue_max, label='Worst')
plt.plot(xaxis, fvalue_median, label='Median')
plt.plot(xaxis, fvalue_min, label='Best')
plt.xlabel('Iteration number')
plt.ylabel('Function value')
plt.xlim(1, iterations)
plt.hlines(opt_value, xaxis[0], xaxis[-1], color='red', label='Optimal value')
plt.legend(loc='upper right')
plt.show()
# Plot function values average of x_avg_exploit: quantiles
qt = np.quantile(fvalue_exploit, [0.25, 0.5, 0.75], axis=0)
plt.figure(figure_counter)
figure_counter += 1
plt.plot(xaxis, qt.T, label=['.25 quantile', 'Median', '.75 quantile'],
         lw=4)
plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80], fontsize=29)
plt.yticks(fontsize=29)
plt.xlabel('Iteration number', fontsize=43)
plt.ylabel('Function value', fontsize=43)
plt.xlim(1, iterations)
plt.hlines(opt_value, xaxis[0], xaxis[-1], color='red',
           label='Optimal value', lw=5)
plt.legend(loc='upper right', fontsize=38)
plt.grid(which='both', axis='y', linestyle='--')
figure = plt.gcf()
figure.set_size_inches(12, 10)
plt.savefig('./Data/{}/{}_{}_quantiles.png'.format(problem, problem, stepsize),
            dpi=100, bbox_inches='tight')
plt.show()
# Plot distance of x_avg evaluated on the surrogates from the global solution
# of the global final surrogate: median, minimum, and maximum
surr_err_median = np.median(surr_err, 0)
surr_err_max = surr_err[np.argmax(np.sum(surr_err, 1)), :]
surr_err_min = surr_err[np.argmin(np.sum(surr_err, 1)), :]
plt.figure(figure_counter)
figure_counter += 1
plt.semilogy(xaxis, surr_err_max, label='worst')
plt.semilogy(xaxis, surr_err_median, label='median')
plt.semilogy(xaxis, surr_err_min, label='best')
plt.legend()
plt.xlabel('Iteration number')
plt.ylabel('Distance from surrogate solution')
plt.show()
# Plot consensus error of x_avg: median, minimum, and maximum
cons_err_median = np.median(cons_err, 0)
cons_err_max = cons_err[np.argmax(np.sum(cons_err, 1)), :]
cons_err_min = cons_err[np.argmin(np.sum(cons_err, 1)), :]
plt.figure(figure_counter)
figure_counter += 1
plt.semilogy(xaxis, cons_err_max, label='worst')
plt.semilogy(xaxis, cons_err_median, label='median')
plt.semilogy(xaxis, cons_err_min, label='best')
plt.legend()
plt.xlabel('Iteration number')
plt.ylabel('Consensus error')
plt.show()
# Plot consensus error of x_avg_exploit: median, minimum, and maximum
cons_err_median = np.median(cons_err_exploit, 0)
cons_err_max = cons_err_exploit[np.argmax(np.sum(cons_err_exploit, 1)), :]
cons_err_min = cons_err_exploit[np.argmin(np.sum(cons_err_exploit, 1)), :]
plt.figure(figure_counter)
figure_counter += 1
plt.semilogy(xaxis, cons_err_max, label='worst')
plt.semilogy(xaxis, cons_err_median, label='median')
plt.semilogy(xaxis, cons_err_min, label='best')
plt.legend(loc='upper right')
plt.xlabel('Iteration number')
plt.ylabel('Consensus error')
plt.show()
