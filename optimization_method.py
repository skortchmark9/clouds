import cvxpy as cp
from cvxpy.atoms.total_variation import tv


# Minimize 1.T @ y subject to y ≥ 0 and T(x) == y
# Possible to omit the tv - could use a lp solver
# basis pursuit

def optimize(x, alpha=0.5):
    # 12747
    altitude_profile = x[0]
    top_down = x[1]

    y = cp.Variable((25, 25, 50), nonneg = True)

    # Objective function
    constraints = [
        cp.mean(cp.mean(y, axis=0), axis=0) == altitude_profile,
        cp.sum(y, axis=2) == top_down,
    ]

    # Objective
    objective_fn = (
        alpha * cp.sum(y) + cp.sum([
            (1 - alpha) * tv(y[i])
            for i in range(25)
        ])
    )

    prob = cp.Problem(cp.Minimize(objective_fn), constraints)
    prob.solve(solver='GUROBI')
    return y.value
