import numpy as np

def grey_wolf_optimizer(fitness_func, lb, ub, dim, n_wolves=10, max_iter=100):
    # Initialize positions of wolves randomly in search space
    wolves = np.random.uniform(lb, ub, (n_wolves, dim))

    # Initialize Alpha, Beta, Delta (best three)
    alpha, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')

    # Main loop
    for t in range(max_iter):
        for i in range(n_wolves):
            # Evaluate fitness
            fitness = fitness_func(wolves[i])

            # Update Alpha, Beta, Delta
            if fitness < alpha_score:
                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = alpha_score, alpha.copy()
                alpha_score, alpha = fitness, wolves[i].copy()
            elif fitness < beta_score:
                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = fitness, wolves[i].copy()
            elif fitness < delta_score:
                delta_score, delta = fitness, wolves[i].copy()

        # Linearly decreasing parameter a
        a = 2 - 2 * (t / max_iter)

        # Update position of each wolf
        for i in range(n_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - wolves[i][j])
                X3 = delta[j] - A3 * D_delta

                # Update current wolf
                wolves[i][j] = (X1 + X2 + X3) / 3

            # Keep wolves within bounds
            wolves[i] = np.clip(wolves[i], lb, ub)

    return alpha, alpha_score
# Example usage
def sphere(x):
    return np.sum(x**2)

best_pos, best_score = grey_wolf_optimizer(sphere, lb=-5, ub=5, dim=5, n_wolves=15, max_iter=100)

print("Best position:", best_pos)
print("Best fitness:", best_score)
