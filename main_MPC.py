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
from utilities import compute_surrogate, normalize_D, denormalize_x
import matlab.engine
# %%`
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
# Create local cost functions
# n = number of variables
n = 5
x = Variable(n)
# Set constraints
lower = np.array([[0.085], [0.1], [10], [-5], [-5]])
upper = np.array([[0.5], [1], [30], [3], [3]])
constr = [x <= 1, x >= -1]
# Compute normalization factors
center = (upper+lower)/2
length = upper-lower
# %%
# Generate random points for the surrogates
N_exp = 10  # Number of points in initial dictionary
N_test = 1  # Number of MonteCarlo repetitions
# Set algorithm parameters
# Number of experiments each agent does with its own IDW
number_outer_iterations_scaled = 1
# Total number of experiments
number_outer_iterations = number_outer_iterations_scaled * nproc
nr_init = 1  # Number of 'swarming' rounds (not used in the paper)
stepsize_list = [0.001]  # Stepsizes to test
num_iterations = 1000  # Number of ADAM iterations
for stepsize in stepsize_list:
    exploration_list = []
    exploitation_list = []
    obj_func_list = []
    dict_list = []
    y_list = []
    scaling_list = []
    for i_test in range(N_test):
        scaling_single_test_list = []
        # Set the current dictionary and create a surrogate
        for idx in range(N_exp):
            x_point = np.array([])
            for jdx in range(n):
                x_point = np.reshape(np.append(x_point,
                                               np.random.uniform(lower[jdx],
                                                                 upper[jdx])),
                                     (1, -1))
            if idx == 0:
                x_dictionary = x_point
            else:
                x_dictionary = np.append(x_dictionary, x_point, axis=0)
        eng1 = matlab.engine.start_matlab()
        x_dictionary_mat = matlab.double(x_dictionary)
        y_dictionary = np.asarray(eng1.benchmark_mpc_calibration(
            x_dictionary_mat, local_rank+1))
        eng1.quit()
        scaling = (max(y_dictionary)-min(y_dictionary))*nproc
        scaling_single_test_list.append(scaling)
        D = {'x': x_dictionary, 'y': y_dictionary}
        Dnorm = normalize_D(D, center, length)
        gpr = compute_surrogate(Dnorm)
        obj_func = 0
        for j in range(N_exp):
            obj_func += gpr.alpha_[j, 0] * Exp(
                SquaredNorm(AffineForm(x, np.identity(n),
                                       -Dnorm['x'][j, :].reshape(-1, 1)))
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
                print('Outer iteration {}'.format(i+1))
            if np.mod(i, nproc) == local_rank:
                turn = True
            # Try different initialization ('swarming')
            final_points = []
            final_global_costs = []
            for j in range(nr_init):
                # ADAM
                x0 = 2*np.random.rand(n, 1) - 1
                adam = GTAdam(agent=agent, initial_condition=x0,
                              enable_log=True)
                adam_seq = adam.run(
                    iterations=num_iterations,
                    stepsize=stepsize,
                    dictionary=Dnorm,
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
                                         denormalize_x(exploration[:, i]
                                                       .reshape(-1, n), center,
                                                       length), axis=0)
                temp = denormalize_x(exploration[:, i].reshape(-1, n),
                                     center, length)
                eng2 = matlab.engine.start_matlab()
                exploration_mat = matlab.double(vector=temp.flatten())
                y_new = np.asarray(eng2.benchmark_mpc_calibration(
                    exploration_mat, local_rank+1))
                eng2.quit()
                y_dictionary = np.append(y_dictionary, y_new).reshape(-1, 1)
                D = {'x': x_dictionary, 'y': y_dictionary}
                Dnorm = normalize_D(D, center, length)
                gpr = compute_surrogate(Dnorm)
                obj_func = 0
                for j in range(N_exp + counter):
                    obj_func += gpr.alpha_[j, 0] * Exp(
                        SquaredNorm(AffineForm(x, np.identity(n),
                                               -Dnorm['x'][j, :].reshape(-1,
                                                                         1)))
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
                x0 = 2*np.random.rand(n, 1) - 1
                adam = GTAdam(agent=agent, initial_condition=x0,
                              enable_log=True)
                adam_seq = adam.run(
                    iterations=num_iterations,
                    stepsize=stepsize,
                    dictionary=Dnorm,
                    local_rank=local_rank,
                    turn=turn,
                    delta=delta,
                    verbose=True)
                achieved_global_cost = adam.run_consensus_on_cost(
                    iterations=100)
                final_points.append(deepcopy(adam.get_result().reshape(n,)))
                final_global_costs.append(float(achieved_global_cost))
            best_solution_idx = np.argmin(final_global_costs)
            exploitation[:, i] = final_points[best_solution_idx]
        exploitation_list.append(exploitation)
        exploration_list.append(exploration)
        scaling_list.append(scaling_single_test_list)
        obj_func_list.append(obj_func)
        dict_list.append(D['x'])
        y_list.append(D['y'])
    # Save data
    with open('./Data/mpc/agents_step_{}.npy'.format(stepsize), 'wb') as f:
        np.save(f, nproc)
        np.save(f, n)
        np.save(f, number_outer_iterations_scaled)
        np.save(f, N_test)
        np.save(f, nr_init)
    np.save('./Data/mpc/agent_{}_sequence_step_{}.npy'.format(agent.id,
                                                              stepsize),
            exploration_list)
    np.save('./Data/mpc/agent_{}_exploitation_step_{}.npy'.format(agent.id,
                                                                  stepsize),
            exploitation_list)
    np.save('./Data/mpc/agent_{}_dictionaries_step_{}.npy'.format(agent.id,
                                                                  stepsize),
            dict_list)
    np.save('./Data/mpc/agent_{}_dictionaries_y_step_{}.npy'.format(agent.id,
                                                                    stepsize),
            y_list)
    np.save('./Data/mpc/agent_{}_scaling_step_{}.npy'.format(agent.id,
                                                             stepsize),
            scaling_list)
