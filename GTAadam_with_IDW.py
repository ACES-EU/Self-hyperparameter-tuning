import numpy as np
from copy import deepcopy
from disropt.agents import Agent
from disropt.algorithms import Algorithm


class GTAdam(Algorithm):
    """
    Args:
        agent (Agent): agent to execute the algorithm
        initial_condition (numpy.ndarray): initial condition for :math:`x_i`
        enable_log (bool): True for enabling log
        beta_1 (float): beta_1
        beta_2 (float): beta_2
        epsilon (float): epsilon

    Attributes:
        agent (Agent): agent to execute the algorithm
        x0 (numpy.ndarray): initial condition
        x (numpy.ndarray): current value of the local solution
        s (numpy.ndarray): current value of the local tracker
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution
        of the (in-)neighbors
        s_neigh (dict): dictionary containing the local tracker
        of the (in-)neighbors
        enable_log (bool): True for enabling log
    """

    def __init__(
        self,
        agent: Agent,
        initial_condition: np.ndarray,
        enable_log: bool = False,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        super(GTAdam, self).__init__(agent, enable_log)
        self.x0 = initial_condition
        self.x = initial_condition
        self.shape = self.x.shape
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.G = 1e6
        # Momenta
        self.g = None
        self.m = None
        self.v = None
        self.s = None
        self.x_neigh = {}
        self.s_neigh = {}
        self.cost_neigh = {}
        self.turn = False
        self.dictionary = None
        self.delta = None

    def iterate_run(
                self,
                stepsize: float,
                projection=False,
                dictionary: dict = {},
                local_rank: int = 0,
                **kwargs):
        """Run a single iterate of the gradient tracking algorithm
        """
        # Point
        data_x = self.agent.neighbors_exchange(self.x)
        for neigh in data_x:
            self.x_neigh[neigh] = data_x[neigh]
        x_kp = self.agent.in_weights[self.agent.id] * self.x
        for j in self.agent.in_neighbors:
            x_kp += self.agent.in_weights[j] * self.x_neigh[j]
        desc = np.divide(self.m, np.sqrt(self.v) + self.epsilon)
        self.x = x_kp - stepsize * desc
        if projection:
            self.x = self.agent.problem.project_on_constraint_set(self.x)
        #####
        # Tracker
        data_s = self.agent.neighbors_exchange(self.s)
        for neigh in data_s:
            self.s_neigh[neigh] = data_s[neigh]
        s_kp = self.agent.in_weights[self.agent.id] * self.s
        for j in self.agent.in_neighbors:
            s_kp += self.agent.in_weights[j] * self.s_neigh[j]
        self.old_g = deepcopy(self.g)
        if self.turn:
            (N_D, b) = np.shape(dictionary['x'])
            d = self.x.T-dictionary['x']
            dn = np.linalg.norm(d, axis=1)
            w = 1/(dn**2+1e-7)
            z = 2/np.pi*np.arctan(1/np.sum(w))
            if z == 0:
                grad_idw = np.zeros(1, b)
            else:
                den = np.zeros((N_D, b))
                for ii in range(np.size(dn)):
                    den[ii, :] = d[ii, :]/(dn[ii]**4+1e-7)
                grad_idw = 4/np.pi/(1+np.sum(w)**2) * np.sum(den, axis=0)
                grad_idw = grad_idw.reshape(b, 1)
            self.g = self.agent.problem.objective_function.subgradient(self.x)
            - self.delta*grad_idw
        else:
            self.g = self.agent.problem.objective_function.subgradient(self.x)
        self.s = s_kp + self.g - self.old_g
        # Update momenta
        self.m = self.beta1 * self.m + (1 - self.beta1) * deepcopy(self.s)
        self.v = np.minimum(
            self.beta2 * self.v + (1 - self.beta2) * deepcopy(self.s) ** 2,
            self.G)

    def run(
        self,
        iterations: int = 1000,
        stepsize: (float, callable) = 0.01,
        dictionary: dict = {},
        local_rank: int = 0,
        turn: bool = False,
        delta: float = 0,
        verbose: bool = False
         ) -> np.ndarray:
        """Run the gradient tracking algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            stepsize: If a float is given as input, the stepsize is constant.
            Default is 0.01.
            verbose: If True print some information during the evolution
            of the algorithm. Defaults to False.

        Raises:
            TypeError: The number of iterations must be an int
            TypeError: The stepsize must be a float

        Returns:
            return the sequence of estimates if enable_log=True.
        """
        self.dictionary = dictionary
        self.turn = turn
        self.delta = delta
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")
        actv_projection = False
        if len(self.agent.problem.constraints) != 0:
            actv_projection = True
        # PRE-star variables allocation
        data = self.agent.neighbors_exchange(self.x)
        for neigh in data:
            self.x_neigh[neigh] = data[neigh]
            self.s_neigh[neigh] = data[neigh]
        if self.turn:
            (N_D, b) = np.shape(dictionary['x'])
            d = self.x.T-dictionary['x']
            dn = np.linalg.norm(d, axis=1)
            w = 1/dn**2
            z = 2/np.pi*np.arctan(1/np.sum(w))
            if z == 0:
                grad_idw = np.zeros(1, b)
            else:
                den = np.zeros((N_D, b))
                for ii in range(np.size(dn)):
                    den[ii, :] = d[ii, :]/(dn[ii]**4)
                grad_idw = 4/np.pi/(1+np.sum(w)**2) * np.sum(den, axis=0)
                grad_idw = grad_idw.reshape(b, 1)
            grad = self.agent.problem.objective_function.subgradient(self.x)
            - self.delta*grad_idw
        else:
            grad = self.agent.problem.objective_function.subgradient(self.x)
        self.g = deepcopy(grad)
        self.s = deepcopy(grad)
        self.m = (1 - self.beta1) * self.g
        self.v = (1 - self.beta2) * self.g ** 2
        #####
        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)
        for k in range(iterations):
            if not isinstance(stepsize, float):
                step = stepsize(k)
            else:
                step = stepsize
            self.iterate_run(stepsize=step, projection=actv_projection,
                             dictionary=dictionary, local_rank=local_rank)
            if self.enable_log:
                self.sequence[k] = self.x
            if verbose:
                if self.agent.id == 0:
                    print("Iteration {}".format(k), end="\r")
        if self.enable_log:
            return self.sequence

    def get_result(self):
        """Return the actual value of x

        Returns:
        numpy.ndarray: value of x
        """
        return self.x

    def get_objective_value(self):
        if not self.turn:
            return self.agent.problem.objective_function.eval(self.x)
        else:
            (N_D, b) = np.shape(self.dictionary['x'])
            d = self.x.T-self.dictionary['x']
            dn = np.linalg.norm(d, axis=1)
            w = 1/dn**2
            z = 2/np.pi*np.arctan(1/np.sum(w))
            return self.agent.problem.objective_function.eval(self.x)
        - self.delta*z

    def run_consensus_on_cost(self, iterations: int = 100):
        local_cost = self.get_objective_value()
        global_cost = local_cost
        for _ in range(iterations):
            data_x = self.agent.neighbors_exchange(global_cost)
            for neigh in data_x:
                self.cost_neigh[neigh] = data_x[neigh]
            global_cost = self.agent.in_weights[self.agent.id] * global_cost
            for j in self.agent.in_neighbors:
                global_cost += self.agent.in_weights[j] * self.cost_neigh[j]
        return global_cost
