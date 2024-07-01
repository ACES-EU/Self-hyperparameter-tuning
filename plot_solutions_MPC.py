import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from utilities import denormalize_x
# %%
# Select the stepsize value related to the results you want to plot
stepsize = 0.001
# %%
# Load all the data for the plots and initialize needed objects
with open('./Data/mpc/agents_step_{}.npy'.format(stepsize), 'rb') as f:
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
dictionaries = []
scaling = []
ylist = []
fvalue = np.zeros((N_test, iterations))
cons_err = np.zeros((N_test, iterations))
surr_err = np.zeros((N_test, iterations))
fvalue_exploit = np.zeros((N_test, iterations))
cons_err_exploit = np.zeros((N_test, iterations))
surr_err_exploit = np.zeros((N_test, iterations))
lower = np.array([[0.085], [0.1], [10], [-5], [-5]])
upper = np.array([[0.5], [1], [30], [3], [3]])
center = (upper+lower)/2
length = upper-lower
for i in range(nproc):
    filename = './Data/mpc/agent_{}_scaling_step_{}.npy'.format(i, stepsize)
    scaling.append(np.load(filename))
    filename = './Data/mpc/agent_{}_sequence_step_{}.npy'.format(i, stepsize)
    x_agent.append(np.load(filename))
    filename = './Data/mpc/agent_{}_exploitation_step_{}.npy'.format(i,
                                                                     stepsize)
    x_agent_exploit.append(np.load(filename))
    filename = './Data/mpc/agent_{}_dictionaries_step_{}.npy'.format(i,
                                                                     stepsize)
    dictionaries.append(np.load(filename))
    filename = './Data/mpc/agent_{}_dictionaries_y_step_{}.npy'.\
        format(i, stepsize)
    ylist.append(np.load(filename))
# Search the lowest objective value among all the dictionary values
ylist_total = ylist[0]
for i in range(nproc-1):
    ylist_total += ylist[i+1]
opt_value = min(ylist_total[0, :, 0])
# Compare the lowest points of the current dictionaries with
# the lowest point known so far for this problem
if opt_value > 0.18268803:
    opt_value = 0.18268803
# %%
# Compute results
# '_exploit' objects refer to points computed without the IDWs
for idx_test in range(N_test):
    x_avg = np.zeros((n, iterations))
    x_avg_exploit = np.zeros((n, iterations))
    for i in range(nproc):
        x_avg += x_agent[i][idx_test]
        x_avg_exploit += x_agent_exploit[i][idx_test]
    x_avg /= nproc
    x_avg_exploit /= nproc
    # Compute function values in x_avg_exploit
    avg_values = np.zeros((iterations, 1))
    x_avg_exploit_denorm = np.copy(x_avg_exploit)
    for i in range(x_avg_exploit.shape[1]):
        x_avg_exploit_denorm[:, i] = denormalize_x(x_avg_exploit[:, i]
                                                   .reshape(-1, n), center,
                                                   length).reshape(5)
    eng1 = matlab.engine.start_matlab()
    x_avg_exploit_mat = matlab.double(x_avg_exploit_denorm.T)
    for i in range(nproc):
        avg_values += np.asarray(eng1.benchmark_mpc_calibration(
            x_avg_exploit_mat, i+1))
    eng1.quit()
    fvalue_exploit[idx_test, :] = avg_values.reshape((1, iterations))
    # Uncomment code below to plot all the single realizations
    # plt.figure(figure_counter)
    # figure_counter += 1
    # plt.semilogy(xaxis, fvalue_exploit[idx_test, :],
    #              label='Proposed method')
    # plt.legend()
    # plt.xlabel('Iteration number')
    # plt.ylabel('Function value')
    # plt.xlim(1, iterations)
    # plt.hlines(opt_value, xaxis[0], xaxis[-1], color='red')
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
    # plt.semilogy(xaxis, c_err, label='Proposed method')
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
plt.plot(xaxis, qt.T, label=['.25 quantile', 'Median', '.75 quantile'], lw=3)
plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Iteration number', fontsize=16)
plt.ylabel('Function value', fontsize=16)
plt.xlim(1, iterations)
plt.hlines(opt_value, xaxis[0], xaxis[-1], color='red', label='Optimal value',
           lw=3)
plt.legend(loc='upper right', fontsize=17)
plt.grid(which='both', axis='y', linestyle='--')
figure = plt.gcf()
figure.set_size_inches(8, 6)
plt.savefig('./Report/Results/mpc_{}_quantiles.png'.format(stepsize), dpi=100)
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
plt.legend()
plt.xlabel('Iteration number')
plt.ylabel('Consensus error')
plt.show()
