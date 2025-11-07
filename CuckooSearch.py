#Cuckoo Search Algorithm to optimize a math function:
import numpy as np
import scipy as sc
# -----------------------------
# Cuckoo Search Algorithm
# -----------------------------
def cuckoo_search(objective_func, n=15, pa=0.25, alpha=0.01, max_iter=100, dim=1, lb=-5, ub=5):
    """
    objective_func : function to minimize
    n              : number of nests (population size)
    pa             : discovery rate (fraction of nests replaced)
    alpha          : step size for Lévy flights
    max_iter       : number of iterations
    dim            : dimension of the problem
    lb, ub         : lower and upper bounds of search space
    """

    # Lévy flight function
    def levy_flight(Lambda):
        sigma1 = (sc.special.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
                 (sc.special.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size=dim)
        v = np.random.normal(0, sigma2, size=dim)
        step = u / (np.abs(v) ** (1 / Lambda))
        return step

    # Initialize nests randomly
    nests = np.random.uniform(lb, ub, (n, dim))
    fitness = np.array([objective_func(x) for x in nests])
    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()

    # Main loop
    for _ in range(max_iter):
        for i in range(n):
            # Generate new solution by Lévy flight
            step = levy_flight(1.5)
            new_nest = nests[i] + alpha * step * (nests[i] - best)
            new_nest = np.clip(new_nest, lb, ub)  # Boundaries
            new_fitness = objective_func(new_nest)

            # Replace if better
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

        # Abandon some nests
        K = np.random.rand(n, dim) > pa
        new_nests = np.random.uniform(lb, ub, (n, dim))
        nests = nests * K + new_nests * (~K)

        # Update fitness and best
        fitness = np.array([objective_func(x) for x in nests])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < objective_func(best):
            best = nests[best_idx].copy()

    return best, objective_func(best)

# example usage:
def func(x):
    return np.sum(x**2)

best_sol, best_val = cuckoo_search(func, dim=1, lb=-10, ub=10, max_iter=100)
print("Best solution:", best_sol)
print("Best value:", best_val)
