import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from GTAadam_with_IDW import GTAdam
from disropt.functions import SquaredNorm, Variable, Exp, AffineForm
from disropt.utils.graph_constructor import binomial_random_graph,\
     metropolis_hastings
from disropt.problems import Problem
import time
from copy import deepcopy
from utilities import f_local, compute_surrogate
# %%
# Get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()
# %%
# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(nproc, p=0.3, seed=1)
W = metropolis_hastings(Adj)
# Reset local seed
np.random.seed(int(time.time())+10*local_rank)
# Create agents
agent = Agent(
    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
    in_weights=W[local_rank, :].tolist(),)
# %%
# Choose the problem to solve. If 'problem' is not among the available ones
# distributed Least Squares will be solved
# n = number of variables
problem = 'brown'
if problem == 'hartman3':
    n = 3
    x = Variable(n)
    constr = [x <= 1, x >= 0]
elif problem == 'rosenbrock':
    n = nproc + 1
    x = Variable(n)
    constr = [x <= 30, x >= -30]
elif problem == 'camel':
    n = 2
    x = Variable(n)
    constr = [x <= 5, x >= -5]
elif problem == 'beale':
    n = 2
    x = Variable(n)
    constr = [x <= 4.5, x >= -4.5]
elif problem == 'brent':
    n = 2
    x = Variable(n)
    constr = [x <= 10, x >= -10]
elif problem == 'brown':
    n = 4
    x = Variable(n)
    constr = [x <= 4, x >= -1]
else:
    n = 4
    M = 100
    x = Variable(n)
    constr = [x <= 1, x >= -1]
    A = np.random.randn(M, n)
    signal = 2*np.random.rand(1, n)-1
    b = np.reshape(np.dot(A, signal.T)/M, (-1, 1))
    A /= M
# %%
# Generate random points for the surrogates
N_exp = 2*n  # Number of points in initial dictionary
N_test = 20  # Number of MonteCarlo repetitions
# Set algorithm parameters
# Number of experiments each agent does with its own IDW
number_outer_iterations_scaled = 20
# Total number of experiments
number_outer_iterations = number_outer_iterations_scaled * nproc
nr_init = 1  # Number of 'swarming' rounds (not used in the paper)
stepsize_list = [0.001]  # Stepsizes to test
num_iterations = 1000  # Number of ADAM iterations
for stepsize in stepsize_list:
    exploration_list = []
    exploitation_list = []
    scaling_list = []
    obj_func_list = []
    dict_list = []
    for i_test in range(N_test):
        scaling_single_test_list = []
        if problem == 'hartman3':
            x_dictionary = np.random.rand(N_exp, n)
            y_dictionary = f_local(x_dictionary, problem,
                                   local_rank=local_rank, nproc=nproc)
        elif problem == 'rosenbrock':
            x_dictionary = 60*np.random.rand(N_exp, n) - 30
            y_dictionary = f_local(x_dictionary, problem,
                                   local_rank=local_rank, nproc=nproc)
        elif problem == 'camel':
            x_dictionary = 10*np.random.rand(N_exp, n) - 5
            y_dictionary = f_local(x_dictionary, problem,
                                   local_rank=local_rank, nproc=nproc)
        elif problem == 'beale':
            x_dictionary = 9*np.random.rand(N_exp, n) - 4.5
            y_dictionary = f_local(x_dictionary, problem,
                                   local_rank=local_rank, nproc=nproc)
        elif problem == 'brent':
            x_dictionary = 20*np.random.rand(N_exp, n) - 10
            y_dictionary = f_local(x_dictionary, problem,
                                   local_rank=local_rank, nproc=nproc)
        elif problem == 'brown':
            x_dictionary = 5*np.random.rand(N_exp, n) - 1
            y_dictionary = f_local(x_dictionary, problem,
                                   local_rank=local_rank, nproc=nproc)
        else:
            x_dictionary = 2*np.random.rand(N_exp, n)-1
            y_dictionary = f_local(x_dictionary, problem, A=A, b=b)
        scaling = (max(y_dictionary)-min(y_dictionary))*nproc
        scaling_single_test_list.append(scaling)
        # Set the current dictionary and create a surrogate
        D = {'x': x_dictionary, 'y': y_dictionary}
        gpr = compute_surrogate(D)
        obj_func = 0
        for j in range(N_exp):
            obj_func += gpr.alpha_[j, 0] * Exp(
                SquaredNorm(AffineForm(x, np.identity(n),
                                       -D['x'][j, :].reshape(-1, 1)))
                / (-2 * gpr.kernel_.get_params()['k2__length_scale'] ** 2))
        obj_func *= gpr.kernel_.get_params()['k1__constant_value']
        # Assign the local cost function
        pb = Problem(obj_func, constr)
        agent.set_problem(pb)
        # Initialize some auxiliary object
        turn = False
        counter = 0
        exploration = np.zeros((n, number_outer_iterations))
        exploitation = np.zeros((n, number_outer_iterations))
        # Run the distributed method
        for i in range(number_outer_iterations):
            delta = 1*scaling  # IDW weight
            if local_rank == 0:
                print('Outer iteration {}'.format(i))
            if np.mod(i, nproc) == local_rank:
                turn = True
            # Try different initialization ('swarming')
            final_points = []
            final_global_costs = []
            for j in range(nr_init):
                # ADAM
                if problem == 'hartman3':
                    x0 = np.random.rand(n, 1)
                elif problem == 'rosenbrock':
                    x0 = 60*np.random.rand(n, 1) - 30
                elif problem == 'camel':
                    x0 = 10*np.random.rand(n, 1) - 5
                elif problem == 'beale':
                    x0 = 9*np.random.rand(n, 1) - 4.5
                elif problem == 'brent':
                    x0 = 20*np.random.rand(n, 1) - 10
                elif problem == 'brown':
                    x0 = 5*np.random.rand(n, 1) - 1
                else:
                    x0 = 2*np.random.rand(n, 1)-1
                adam = GTAdam(agent=agent, initial_condition=x0,
                              enable_log=True)
                adam_seq = adam.run(
                    iterations=num_iterations,
                    stepsize=stepsize,
                    dictionary=D,
                    local_rank=local_rank,
                    turn=turn,
                    delta=delta,
                    verbose=True)
                # Find global cost via consensus
                achieved_global_cost = \
                    adam.run_consensus_on_cost(iterations=100)
                final_points.append(deepcopy(adam.get_result().reshape(n,)))
                final_global_costs.append(float(achieved_global_cost))
            # Pick the solution leading to the lower cost
            best_solution_idx = np.argmin(final_global_costs)
            exploration[:, i] = final_points[best_solution_idx]
            # In a cyclic way each agent updates its dictionary and surrogate
            if turn:
                counter += 1
                x_dictionary = np.append(x_dictionary,
                                         exploration[:, i].reshape(-1, n),
                                         axis=0)
                if (problem == 'hartman3') or (problem == 'rosenbrock') \
                   or (problem == 'camel') or (problem == 'beale') \
                   or (problem == 'brent') or (problem == 'brown'):
                    y_new = f_local(exploration[:, i].reshape(-1, n), problem,
                                    local_rank=local_rank, nproc=nproc)
                else:
                    y_new = f_local(exploration[:, i].reshape(-1, n), problem,
                                    A=A, b=b)
                y_dictionary = np.append(y_dictionary, y_new, axis=0)
                D = {'x': x_dictionary, 'y': y_dictionary}
                gpr = compute_surrogate(D)
                obj_func = 0
                for j in range(N_exp + counter):
                    obj_func += gpr.alpha_[j, 0] * Exp(
                        SquaredNorm(AffineForm(x, np.identity(n),
                                               -D['x'][j, :].reshape(-1, 1)))
                        / (-2*gpr.kernel_.get_params()['k2__length_scale']**2))
                obj_func *= gpr.kernel_.get_params()['k1__constant_value']
                pb = Problem(obj_func, constr)
                agent.set_problem(pb)
                scaling = (max(y_dictionary)-min(y_dictionary))*nproc
                scaling_single_test_list.append(scaling)
                turn = False
            # Compute solutions without IDW, for evaluation purposes
            final_points = []
            final_global_costs = []
            for j in range(nr_init):
                if problem == 'hartman3':
                    x0 = np.random.rand(n, 1)
                elif problem == 'rosenbrock':
                    x0 = 60*np.random.rand(n, 1) - 30
                elif problem == 'camel':
                    x0 = 10*np.random.rand(n, 1) - 5
                elif problem == 'beale':
                    x0 = 9*np.random.rand(n, 1) - 4.5
                elif problem == 'brent':
                    x0 = 20*np.random.rand(n, 1) - 10
                elif problem == 'brown':
                    x0 = 5*np.random.rand(n, 1) - 1
                else:
                    x0 = 2*np.random.rand(n, 1)-1
                adam = GTAdam(agent=agent, initial_condition=x0,
                              enable_log=True)
                adam_seq = adam.run(
                    iterations=num_iterations,
                    stepsize=stepsize,
                    dictionary=D,
                    local_rank=local_rank,
                    turn=turn,
                    delta=delta,
                    verbose=True)
                achieved_global_cost = \
                    adam.run_consensus_on_cost(iterations=100)
                final_points.append(deepcopy(adam.get_result().reshape(n,)))
                final_global_costs.append(float(achieved_global_cost))
            best_solution_idx = np.argmin(final_global_costs)
            exploitation[:, i] = final_points[best_solution_idx]
        exploitation_list.append(exploitation)
        exploration_list.append(exploration)
        scaling_list.append(scaling_single_test_list)
        obj_func_list.append(obj_func)
        dict_list.append(D['x'])
    # Save data
    with open('./Data/{}/agents_step_{}.npy'.format(problem, stepsize), 'wb') \
         as f:
        np.save(f, nproc)
        np.save(f, n)
        np.save(f, number_outer_iterations_scaled)
        np.save(f, N_test)
        np.save(f, nr_init)
    np.save('./Data/{}/agent_{}_sequence_step_{}.npy'.format(problem, agent.id,
                                                             stepsize),
            exploration_list)
    np.save('./Data/{}/agent_{}_exploitation_step_{}.npy'.format(problem,
                                                                 agent.id,
                                                                 stepsize),
            exploitation_list)
    np.save('./Data/{}/agent_{}_dictionaries_step_{}.npy'.format(problem,
                                                                 agent.id,
                                                                 stepsize),
            dict_list)
    np.save('./Data/{}/agent_{}_scaling_step_{}.npy'.format(problem, agent.id,
                                                            stepsize),
            scaling_list)
    with open('./Data/{}/agent_{}_function_step_{}.pkl'.format(problem,
                                                               agent.id,
                                                               stepsize),
              'wb') as output:
        pickle.dump(obj_func_list, output, pickle.HIGHEST_PROTOCOL)
    with open('./Data/{}/constraints_step_{}.pkl'.format(problem, stepsize),
              'wb') as output:
        pickle.dump(constr, output, pickle.HIGHEST_PROTOCOL)
    if problem == 'ls':
        np.save('./Data/{}/agent_{}_A_step_{}.npy'.format(problem, agent.id,
                                                          stepsize), A)
        np.save('./Data/{}/agent_{}_b_step_{}.npy'.format(problem, agent.id,
                                                          stepsize), b)
