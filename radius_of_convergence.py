import numpy as np
import jax.numpy as jnp
import jax
from jax import grad
import sympy

x = sympy.Symbol('x')

def newtons_method(func, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        # Update x using Newton's method
        x = x - func(x) / jax.grad(func)(x)
        # Check for convergence
        if jnp.abs(func(x)) < tol:
            return x
    return x

def converges(func, solution, x0):
    """
    Check if Newton's method converges to the solution starting from x0.

    Parameters:
    func (function): The function for which to find the root.
    solution (float): The expected solution.
    x0 (float): The initial guess.

    Returns:
    bool: True if Newton's method converges to the solution, False otherwise.
    """
    return np.isclose(float(newtons_method(func, x0)), float(solution))

def radius_of_convergence_bisection(func, solutions, max_x=100., max_iter=100):
    """
    Calculate the radius of convergence for each solution using the bisection method.

    Parameters:
    func (function): The function for which to find the radius of convergence.
    solutions (list): List of known solutions.
    max_x (float): The maximum x value to consider.
    max_iter (int): The maximum number of iterations for the bisection method.

    Returns:
    list: A list of tuples representing the lower and upper radius of convergence for each solution.
    """
    radii_of_convergence = []
    for i, solution in enumerate(solutions):
        min_radius = float(solution)
        max_radius = float(solutions[i + 1]) if len(solutions) > (i + 1) else max_x

        if converges(func, solution, max_radius):
            upper_radius = None
        elif not converges(func, solution, min_radius):
            upper_radius = 0.0
        else:
            upper_radius = bisection_method(func, solution, min_radius, max_radius, max_iter)

        min_radius = float(solution)
        max_radius = float(solutions[i - 1]) if i > 0 else -max_x

        if converges(func, solution, max_radius):
            lower_radius = None
        elif not converges(func, solution, min_radius):
            lower_radius = 0.0
        else:
            lower_radius = bisection_method(func, solution, min_radius, max_radius, max_iter)

        radii_of_convergence.append((lower_radius, upper_radius))
    return radii_of_convergence

def bisection_method(func, solution, min_radius, max_radius, max_iter):
    """
    Use the bisection method to find the radius of convergence.

    Parameters:
    func (function): The function for which to find the radius of convergence.
    solution (float): The known solution.
    min_radius (float): The minimum radius to consider.
    max_radius (float): The maximum radius to consider.
    max_iter (int): The maximum number of iterations.

    Returns:
    float: The radius of convergence.
    """
    midpoint = (min_radius + max_radius) / 2
    for _ in range(max_iter):
        if converges(func, solution, midpoint):
            min_radius = midpoint
        else:
            max_radius = midpoint
        midpoint = (min_radius + max_radius) / 2
        if np.abs(midpoint - solution) <= 1e-6 or np.abs(midpoint - max_radius) <= 1e-6:
            break
    return np.round(midpoint, 5)

def generate_random_polynomial(trig=False, max_degree=4):
    """
    Generates a random function as a linear combination of x, x^2, x^3, x^4, sin(x), and cos(x).

    Parameters:
    trig (bool): Whether to include trigonometric functions (sin and cos) in the basis.
    max_degree (int): The maximum degree of the polynomial terms.

    Returns:
    tuple: A tuple containing the generated function and its symbolic expression.
    """
    # Randomly choose the degree of the function
    degree = np.random.choice(np.arange(max_degree), p=np.ones(max_degree) / max_degree)
    
    # Create the basis functions up to the chosen degree
    basis = [1] + [x**d for d in range(1, degree + 2)]
    
    # Optionally add trigonometric functions to the basis
    if trig:
        if np.random.rand() < 0.5:
            basis.append(sympy.sin(x))
        if np.random.rand() < 0.5:
            basis.append(sympy.cos(x))
    
    # Generate random coefficients for the basis functions
    coefficients = np.random.normal(0, 5, len(basis))
    
    # Create the random expression by summing the basis functions with their coefficients
    random_expr = sum(c * b for c, b in zip(coefficients, basis))
    
    # Convert the symbolic expression to a JAX-compatible function
    func = lambda x_val: jnp.array(sympy.lambdify(x, random_expr, 'numpy')(x_val))
    
    return func, random_expr

def generate_random_trig(phase_shift=False):
    """
    Generates a random function as a linear combination of sin(x) and cos(x).

    Returns:
    tuple: A tuple containing the generated function and its symbolic expression.
    """
    # Randomly choose the coefficients for sin(x) and cos(x)
    coefs = np.round(np.random.normal(0, 5, 4), 2)
    beta = np.round(np.random.uniform(-2,2), 2)

    phase = 0
    if phase_shift:
        phase = np.random.uniform(0, 2 * np.pi)
    
    # Create the random expression as a linear combination of sin(x) and cos(x)
    random_expr = coefs[0] * sympy.sin(coefs[1]*x + phase) + coefs[2] * sympy.cos(coefs[3]* x) + beta
    
    # Convert the symbolic expression to a JAX-compatible function
    func = lambda x_val: jnp.array(sympy.lambdify(x, random_expr, 'numpy')(x_val))
    
    # Compute the period of the function
    gcd = sympy.gcd(int(coefs[1]*100), int(coefs[3]*100))
    p = coefs[1] * 100 / gcd
    q = coefs[3] * 100 / gcd
    period = abs((q / coefs[1]) * 2 * np.pi)

    return func, random_expr, period

def analyze_function(func, expr):
    """
    Analyzes a function by finding its roots and determining the radius of convergence.

    Parameters:
    func (function): The function to analyze.
    expr (sympy.Expr): The symbolic expression of the function.

    Returns:
    tuple: A tuple containing the symbolic expression and the list of real roots.
    """
    print(f"Generated function: {expr}")
    
    # Find the real roots of the symbolic expression
    roots = sympy.solve(expr, x)
    real_roots = [r.evalf() for r in roots if r.is_real]
    real_roots.sort()
    print(f"Real roots: {real_roots}")
    
    # Apply radius of convergence analysis if there are real roots
    if real_roots:
        convergence_data = radius_of_convergence_bisection(func, real_roots)
        print(f"Radius of convergence data: {convergence_data}")
    else:
        print("No real roots found, skipping convergence analysis.")
    
    return expr, real_roots

def deriv_of_order(func, order):
    """
    Compute the derivative of a given order for a function.

    Parameters:
    func (function): The function to differentiate.
    order (int): The order of the derivative.

    Returns:
    function: The derivative function of the specified order.
    """
    output = func
    for i in range(order):
        output = grad(output)
    return output

def taylor_approx(func, point=0.0, order=3):
    """
    Compute the Taylor series approximation of a function at a given point.

    Parameters:
    func (function): The function to approximate.
    point (float): The point at which to approximate the function.
    order (int): The order of the Taylor series.

    Returns:
    sympy.Expr: The Taylor series approximation of the function.
    """
    approx = 0
    for i in range(order + 1):
        # Compute the i-th derivative at the given point
        derivative_at_point = deriv_of_order(func, i)(point)
        # Add the i-th term of the Taylor series to the approximation
        approx += derivative_at_point / sympy.factorial(i) * (x - point)**i
    return approx