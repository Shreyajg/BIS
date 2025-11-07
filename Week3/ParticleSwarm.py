import numpy as np
def particle_swarm_optimizer(fitness_func, lb, ub, dim, n_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
    # Initialize particles
    X = np.random.uniform(lb, ub, (n_particles, dim))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_fitness = np.apply_along_axis(fitness_func, 1, X)
    gbest_idx = np.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fitness[gbest_idx]

    for _ in range(max_iter):
        r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
        X += V
        X = np.clip(X, lb, ub)
        fitness = np.apply_along_axis(fitness_func, 1, X)

        # Update personal & global bests
        better = fitness < pbest_fitness
        pbest[better] = X[better]
        pbest_fitness[better] = fitness[better]
        if pbest_fitness.min() < gbest_fit:
            gbest_fit = pbest_fitness.min()
            gbest = pbest[np.argmin(pbest_fitness)].copy()

    return gbest, gbest_fit
def sphere(x): return np.sum(x**2)
best_pos, best_fit = particle_swarm_optimizer(sphere, -5, 5, 3)
print("best pos: ",best_pos)
print("best fit: ", best_fit)