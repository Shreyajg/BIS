import numpy as np

def ant_colony_optimizer(fitness_func, lb, ub, dim, n_ants=20, archive_size=40, max_iter=100, q=0.5, xi=0.85):
    # Initialize archive (solutions + fitness)
    archive = np.random.uniform(lb, ub, (archive_size, dim))
    fitness = np.apply_along_axis(fitness_func, 1, archive)
    
    for _ in range(max_iter):
        # Sort archive by fitness
        sorted_idx = np.argsort(fitness)
        archive = archive[sorted_idx]
        fitness = fitness[sorted_idx]

        # Compute weights (Gaussian kernel)
        w = (1 / (np.sqrt(2 * np.pi) * q * archive_size)) * np.exp(-((np.arange(archive_size))**2) / (2 * (q**2) * archive_size**2))
        w /= np.sum(w)

        # Build new solutions
        new_ants = np.zeros((n_ants, dim))
        for i in range(n_ants):
            idx = np.random.choice(archive_size, p=w)
            mean = archive[idx]
            sigma = np.abs(archive - mean).mean(axis=0) * xi
            new_ants[i] = mean + np.random.randn(dim) * sigma

        # Evaluate new ants
        new_ants = np.clip(new_ants, lb, ub)
        new_fitness = np.apply_along_axis(fitness_func, 1, new_ants)

        # Merge and keep best
        archive = np.vstack((archive, new_ants))
        fitness = np.concatenate((fitness, new_fitness))
        idx = np.argsort(fitness)[:archive_size]
        archive, fitness = archive[idx], fitness[idx]

    return archive[0], fitness[0]
def sphere(x): return np.sum(x**2)
best_pos, best_fit = ant_colony_optimizer(sphere, -5, 5, 3)
print("best pos: ",best_pos)
print("best fit: ", best_fit)
