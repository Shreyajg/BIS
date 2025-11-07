import numpy as np
def genetic_algorithm(fitness_func, lb, ub, dim, pop_size=30, max_iter=100, mutation_rate=0.1, crossover_rate=0.8):
    # Initialize population
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.apply_along_axis(fitness_func, 1, pop)

    for _ in range(max_iter):
        # Selection (tournament)
        parents = []
        for _ in range(pop_size):
            i, j = np.random.randint(0, pop_size, 2)
            parents.append(pop[i] if fitness[i] < fitness[j] else pop[j])
        parents = np.array(parents)

        # Crossover
        offspring = parents.copy()
        for i in range(0, pop_size, 2):
            if np.random.rand() < crossover_rate:
                cross_point = np.random.randint(1, dim)
                offspring[i, :cross_point], offspring[i+1, :cross_point] = parents[i+1, :cross_point], parents[i, :cross_point]

        # Mutation
        for i in range(pop_size):
            for j in range(dim):
                if np.random.rand() < mutation_rate:
                    offspring[i, j] = np.random.uniform(lb, ub)

        offspring = np.clip(offspring, lb, ub)
        new_fitness = np.apply_along_axis(fitness_func, 1, offspring)

        # Elitism: keep best
        combined = np.vstack((pop, offspring))
        combined_fit = np.concatenate((fitness, new_fitness))
        idx = np.argsort(combined_fit)[:pop_size]
        pop, fitness = combined[idx], combined_fit[idx]

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx]
def sphere(x): return np.sum(x**2)
best_pos, best_fit = genetic_algorithm(sphere, -5, 5, 3)
print("best pos: ",best_pos)
print("best fit: ", best_fit)