import numpy as np

def parallel_cellular_optimizer(fitness_func, lb, ub, dim, grid_size=(5, 5), 
                                max_iter=100, step_size=0.3):
    """
    Parallel Cellular Optimization Algorithm (PCOA)
    Minimizes the given fitness function.

    Parameters:
        fitness_func : function -> The objective function to minimize.
        lb, ub        : float   -> Lower and upper bounds.
        dim           : int     -> Dimensionality of the problem.
        grid_size     : (int,int)-> Grid dimensions (rows, cols).
        max_iter      : int     -> Number of iterations.
        step_size     : float   -> Step size λ (controls local movement).
    """

    rows, cols = grid_size
    # Initialize 2D grid of random solutions
    grid = np.random.uniform(lb, ub, (rows, cols, dim))
    fitness = np.apply_along_axis(fitness_func, 2, grid)

    # Define Moore neighborhood (8 neighbors)
    def get_neighbors(i, j):
        neighbors = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == 0 and y == 0:
                    continue
                ni, nj = (i + x) % rows, (j + y) % cols  # wrap around (toroidal grid)
                neighbors.append((ni, nj))
        return neighbors

    # Main loop
    for _ in range(max_iter):
        new_grid = grid.copy()

        # Parallel-like update (all cells use current grid state)
        for i in range(rows):
            for j in range(cols):
                # Find best neighbor
                neighbors = get_neighbors(i, j)
                best_neighbor = min(neighbors, key=lambda n: fitness[n])
                best_sol = grid[best_neighbor]
                best_fit = fitness[best_neighbor]

                # Generate new candidate
                candidate = grid[i, j] + step_size * np.random.rand() * (best_sol - grid[i, j])
                candidate = np.clip(candidate, lb, ub)
                candidate_fit = fitness_func(candidate)

                # Replace if better
                if candidate_fit < fitness[i, j]:
                    new_grid[i, j] = candidate
                    fitness[i, j] = candidate_fit

        grid = new_grid.copy()

    # Return global best
    best_idx = np.unravel_index(np.argmin(fitness), fitness.shape)
    best_sol = grid[best_idx]
    best_fit = fitness[best_idx]
    return best_sol, best_fit
# Example: Sphere function (minimum at 0)
def sphere(x):
    return np.sum(x**2)

best_sol, best_fit = parallel_cellular_optimizer(
    sphere, lb=-5, ub=5, dim=3, grid_size=(6, 6), max_iter=100
)

print("Best solution:", best_sol)
print("Best fitness:", best_fit)
