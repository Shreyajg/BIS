import operator, random
import numpy as np

ops = [('+', operator.add), ('-', operator.sub), ('*', operator.mul)]
terms = ['x', 1, 2, 3]

def random_expr(depth=2):
    if depth == 0:
        return str(random.choice(terms))
    op, _ = random.choice(ops)
    return f"({random_expr(depth-1)} {op} {random_expr(depth-1)})"

def fitness(expr, target_func, samples=np.linspace(-5,5,20)):
    try:
        errors = []
        for x in samples:
            y_pred = eval(expr)
            y_true = target_func(x)
            errors.append((y_true - y_pred)**2)
        return np.mean(errors)
    except Exception:
        return float('inf')

def gene_expression_programming(target_func, pop_size=30, max_iter=100):
    pop = [random_expr(3) for _ in range(pop_size)]
    for _ in range(max_iter):
        fit = [fitness(expr, target_func) for expr in pop]
        new_pop = []
        for _ in range(pop_size//2):
            i, j = np.argsort(fit)[:2]
            p1, p2 = pop[i], pop[j]
            cross_point = random.randint(1, len(p1)-1)
            child = p1[:cross_point] + p2[cross_point:]
            if random.random() < 0.2:
                child = random_expr(3)
            new_pop.extend([child])
        pop.extend(new_pop)
        pop = sorted(pop, key=lambda e: fitness(e, target_func))[:pop_size]
    return pop[0]
# Target function: y = x^2
best_expr = gene_expression_programming(lambda x: x**2)
print("Best evolved expression:", best_expr)
