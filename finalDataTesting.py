import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap, scipy
import sympy
import matplotlib.pyplot as plt

x = sympy.Symbol('x')
e = jnp.e


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
    return jnp.round(float(newtons_method(func, x0)), 6) == jnp.round(float(solution), 6)


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
    degree = np.random.choice(jnp.arange(max_degree), p=jnp.ones(max_degree) / max_degree)

    # Create the basis functions up to the chosen degree
    basis = [1] + [x ** d for d in range(1, degree + 2)]

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
    coefs = jnp.round(np.random.normal(0, 5, 4), 2)
    beta = jnp.round(np.random.uniform(-2, 2), 2)

    phase = 0
    if phase_shift:
        phase = np.random.uniform(0, 2 * jnp.pi)

    # Create the random expression as a linear combination of sin(x) and cos(x)
    random_expr = coefs[0] * sympy.sin(coefs[1] * x + phase) + coefs[2] * sympy.cos(coefs[3] * x) + beta

    # Convert the symbolic expression to a JAX-compatible function
    func = lambda x_val: jnp.array(sympy.lambdify(x, random_expr, 'numpy')(x_val))

    # Compute the period of the function
    gcd = sympy.gcd(int(coefs[1] * 100), int(coefs[3] * 100))
    p = coefs[1] * 100 / gcd
    q = coefs[3] * 100 / gcd
    period = abs((q / coefs[1]) * 2 * jnp.pi)

    return func, random_expr, period


def generate_random_combined(trig=False, max_degree=4, phase_shift=False):
    """
    Generates a random function as a linear combination of polynomial terms (x^1 to x^max_degree) and
    trigonometric functions (sin(x), cos(x)).

    Parameters:
    trig (bool): Whether to include trigonometric functions (sin and cos) in the basis.
    max_degree (int): The maximum degree of the polynomial terms.
    phase_shift (bool): Whether to apply a phase shift to the sine and cosine terms.

    Returns:
    tuple: A tuple containing the generated function and its symbolic expression.
    """
    # degree choice
    degree = np.random.choice(jnp.arange(max_degree), p=jnp.ones(max_degree) / max_degree)

    # polynomial basis
    basis = [1] + [x ** d for d in range(1, degree + 2)]

    # trig parts
    if trig:
        coefs = jnp.round(np.random.normal(0, 5, 4), 2)
        beta = jnp.round(np.random.uniform(-2, 2), 2)
        phase = 0
        if phase_shift:
            phase = np.random.uniform(0, 2 * jnp.pi)

        # Add sine and cosine terms with random coefficients
        if np.random.rand() < 0.5:
            basis.append(coefs[0] * sympy.sin(coefs[1] * x + phase))
        if np.random.rand() < 0.5:
            basis.append(coefs[2] * sympy.cos(coefs[3] * x))

    # Generate random coefficients for all the basis functions
    coefficients = np.random.normal(0, 5, len(basis))

    # Create the random expression by summing the basis functions with their coefficients
    random_expr = sum(c * b for c, b in zip(coefficients, basis))

    # Convert the symbolic expression to a JAX-compatible function
    func = lambda x_val: jnp.array(sympy.lambdify(x, random_expr, 'numpy')(x_val))

    # If trigonometric terms were added, compute the period of the function
    period = None
    if trig:
        # Compute the period of the sine and cosine terms
        gcd = sympy.gcd(int(coefs[1] * 100), int(coefs[3] * 100))
        p = coefs[1] * 100 / gcd
        q = coefs[3] * 100 / gcd
        period = abs((q / coefs[1]) * 2 * jnp.pi)

    return func, random_expr, period


def analyze_function(func, expr, real_roots=None):
    """
    Analyzes a function by finding its roots and determining the radius of convergence.

    Parameters:
    func (function): The function to analyze.
    expr (sympy.Expr): The symbolic expression of the function.

    Returns:
    tuple: A tuple containing the symbolic expression and the list of real roots.
    """
    print(f"Generated function: {expr}")

    # If not already given, find the real roots of the symbolic expression
    if real_roots is None:
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


def taylor_approx(func, point=0.0, order=3, plot=False):
    """
    Compute the Taylor series approximation of a function at a given point and optionally plot it.

    Parameters:
    func (function): The function to approximate.
    point (float): The point at which to approximate the function.
    order (int): The order of the Taylor series.
    plot (bool): Whether to plot the original function and its Taylor approximation.

    Returns:
    sympy.Expr: The Taylor series approximation of the function.
    """
    approx = 0
    for i in range(order + 1):
        # Compute the i-th derivative at the given point
        derivative_at_point = deriv_of_order(func, i)(point)
        # Add the i-th term of the Taylor series to the approximation
        approx += derivative_at_point / sympy.factorial(i) * (x - point) ** i

    if plot:
        x_vals = np.linspace(point - 5, point + 5, 400)
        y_vals = np.array([func(x) for x in x_vals])
        taylor_func = sympy.lambdify(sympy.Symbol('x'), approx, 'numpy')
        taylor_vals = np.array([taylor_func(x) for x in x_vals])

        plt.plot(x_vals, y_vals, label="Original function")
        plt.plot(x_vals, taylor_vals, label=f"Taylor series (order={order})")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Original Function vs. Taylor Series Approximation")
        plt.show()

    return approx


def truncated_fft(function, order=3, point=0.0, N=int(1e6), ds=1e-3, plot=False):
    """
    Truncates the FFT of the input data to the specified number of terms and
    returns a SymPy expression for the truncated Fourier series.

    Args:
        function: The input function (a Python/NumPy/JAX function).
        n_terms: The number of terms to keep.
        N: The number of points in the input data.
        ds: The spacing between points in the input data.
        plot: Boolean, whether to plot the original and reconstructed data.

    Returns:
        Sympy expression of data
        Reconstructed data
    """
    T = N * ds
    x_np = np.linspace(point - T / 2, point + T / 2, N, endpoint=False)  # NumPy array for FFT
    x_sp = sympy.Symbol('x')
    data = function(x_np)  # Evaluate the function using NumPy

    # Check if all values in data are real
    if np.any(np.isnan(data)):
        nan_indices = np.where(np.isnan(data))[0]
        nan_points = x_np[nan_indices]
        raise ValueError(f"Function evaluation returned NaN for points in x_np: {nan_points}. "
                         f"Largest value: {np.max(nan_points)}, Smallest value: {np.min(nan_points)}")

    fft_result = np.fft.fft(data)
    truncated_fft = np.zeros_like(fft_result, dtype=complex)

    truncated_fft[:order] = fft_result[:order]
    truncated_fft[-order:] = fft_result[-order:]

    reconstructed_data = np.fft.ifft(truncated_fft)

    f0 = 1 / T
    a0 = 2 * fft_result[0].real / N
    fourier_series_expr = a0 / 2

    for k in range(1, order + 1):
        ak_complex = 2 * fft_result[k] / N
        ak = ak_complex.real
        bk = -ak_complex.imag
        fourier_series_expr += ak * sympy.cos(2 * sympy.pi * k * f0 * x_sp) + bk * sympy.sin(
            2 * sympy.pi * k * f0 * x_sp)

    if plot:
        plt.plot(x_np, data, label="Original data")
        plt.plot(x_np, reconstructed_data.real, label=f"Reconstructed data (n={order})")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Original vs. Reconstructed Data")
        plt.show()

    return fourier_series_expr, reconstructed_data.real

#==========Function 1==========
def f1(x):
    return jnp.exp(x) - 1
f1_expr = sympy.exp(x) - 1
f1_fourier_expr, _ = truncated_fft(f1, order=3, point=0.0, N=50, ds=0.1, plot=False)
f1_fourier = sympy.lambdify(x, f1_fourier_expr, 'jax')

f1_taylor_expr = taylor_approx(f1, point=0.0, order=4, plot=False)
f1_taylor = sympy.lambdify(x, f1_taylor_expr, 'jax')


analyze_function(f1, f1_expr)
analyze_function(f1_taylor, f1_taylor_expr) 
analyze_function(f1_fourier, f1_fourier_expr, real_roots=[0.31056, 2.49824])

#==========Function 2==========
def f2(x):
    return jnp.exp(2 * x) - 2
f2_expr = sympy.exp(2 * x) - 2
f2_fourier_expr, _ = truncated_fft(f2, order=3, point=0.0, N=50, ds=0.1, plot=False)
f2_fourier = sympy.lambdify(x, f2_fourier_expr, 'jax')

f2_taylor_expr = taylor_approx(f2, point=0.0, order=4, plot=False)
f2_taylor = sympy.lambdify(x, f2_taylor_expr, 'jax')

analyze_function(f2, f2_expr)
analyze_function(f2_taylor, f2_taylor_expr) 
analyze_function(f2_fourier, f2_fourier_expr, real_roots=[-1.37006, 0.42322])

#==========Function 3==========
def f3(x):
    return jnp.exp(3 * x) + 4 * x
f3_expr = sympy.exp(3 * x) + 4 * x
f3_fourier_expr, _ = truncated_fft(f3, order=3, point=0.0, N=50, ds=0.1, plot=False)
f3_fourier = sympy.lambdify(x, f3_fourier_expr, 'jax')
#print(f3_fourier_expr)
f3_taylor_expr = taylor_approx(f3, point=0.0, order=4, plot=False)
f3_taylor = sympy.lambdify(x, f3_taylor_expr, 'jax')

analyze_function(f3, f3_expr)
analyze_function(f3_taylor, f3_taylor_expr) 
analyze_function(f3_fourier, f3_fourier_expr, real_roots=[-1.17244, 0.46482])

# WORKING EXAMPLE
#==========Function 4==========
def f4(x):
    return jnp.log(x) # Define a function using jnp
f4_expr = sympy.log(x) # Define its symbolic notation

# First issue: originally, we were evaluating at ln(0), which blows up. Not has been changed to 2.51, and the interval is now 2.51 +/- 50*0.1, so (0.01, 5.01)
f4_fourier_expr, _ = truncated_fft(f4, order=3, point=2.51, N=50, ds=0.1, plot=False) # Your interval is point +/- N*ds. This must be a continuous part of the function
f4_fourier = sympy.lambdify(x, f4_fourier_expr, 'jax') # Convert the symbolic expression to a jax function, not numpy

f4_taylor_expr = taylor_approx(f4, point=2.51, order=4, plot=False) # This is the taylor approximation, also has plotting functionality now
f4_taylor = sympy.lambdify(x, f4_taylor_expr, 'jax') # Same process as above


analyze_function(f4, f4_expr) # This prints out the roots of the function, and correct intervals (I believe, it looks like it's working)
analyze_function(f4_taylor, f4_taylor_expr) # This prints out the roots of the taylor approximation
# HERE WAS SECOND ISSUE: sympy can't solve roots of fourier series. Just copy paste the sympy expression into desmos, set p = 1, i = pi (so that the constants work out),
# and find the roots manually. This is probably the easiest way to do it.
analyze_function(f4_fourier, f4_fourier_expr, real_roots=[-0.1698, 0.95926])

#==========Function 5==========
def f5(x):
    return jnp.log(x + 1)
f5_expr = sympy.log(x + 1, e)
f5_fourier_expr, _ = truncated_fft(f5, order=3, point=1.51, N=50, ds=0.1, plot=True)
f5_fourier = sympy.lambdify(x, f5_fourier_expr, 'jax')

f5_taylor_expr = taylor_approx(f5, point=1.51, order=4, plot=True)
f5_taylor = sympy.lambdify(x, f5_taylor_expr, 'jax')

analyze_function(f5, f5_expr)
analyze_function(f5_taylor, f5_taylor_expr) 
analyze_function(f5_fourier, f5_fourier_expr, real_roots=[-0.1698, 0.95926])

#==========Function 6==========
def f6(x):
    return -jnp.log(x)
f6_expr = -sympy.log(x, e)
f6_fourier_expr, _ = truncated_fft(f6, order=3, point=2.51, N=50, ds=0.1, plot=False)
f6_fourier = sympy.lambdify(x, f6_fourier_expr, 'jax')

f6_taylor_expr = taylor_approx(f6, point=2.51, order=4, plot=False)
f6_taylor = sympy.lambdify(x, f6_taylor_expr, 'jax')

analyze_function(f6, f6_expr)
analyze_function(f6_taylor, f6_taylor_expr) 
analyze_function(f6_fourier, f6_fourier_expr, real_roots=[-0.1698, 0.95926])

#==========Function 7==========
def f7(x):
    return jnp.log(x - 1) - 3 * x + 6
f7_expr = sympy.log(x - 1, e) - 3 * x + 6
f7_fourier_expr, _ = truncated_fft(f7, order=3, point=2.1, N=10, ds=0.1, plot=False)
f7_fourier = sympy.lambdify(x, f7_fourier_expr, 'jax')

f7_taylor_expr = taylor_approx(f7, point=2.1, order=4, plot=False)
f7_taylor = sympy.lambdify(x, f7_taylor_expr, 'jax')

analyze_function(f7, f7_expr)
analyze_function(f7_taylor, f7_taylor_expr) 
analyze_function(f7_fourier, f7_fourier_expr, real_roots=[-0.042659, 0.183706])

# Quick loops with all the functions, INTERVALS MIGHT BE OFF, i didn't check them
functions = [
    (lambda x: jnp.exp(x) - 1, sympy.exp(x) - 1),
    (lambda x: jnp.exp(2*x) - 2, sympy.exp(2*x) - 2),
    (lambda x: jnp.exp(3*x) + 4*x, sympy.exp(3*x) + 4*x),
    (lambda x: jnp.log(x), sympy.log(x)),
    (lambda x: jnp.log(x + 1), sympy.log(x + 1)),
    (lambda x: -jnp.log(x), -sympy.log(x)),
    (lambda x: jnp.log(x) + 3*x, sympy.log(x) + 3*x),
    (lambda x: jnp.log(x) + jnp.exp(x), sympy.log(x) + sympy.exp(x))
]

'''
# Process each function
for i, (func, func_expr) in enumerate(functions, start=1):
    print(f"Processing function {i}")

    # Fourier series approximation
    fourier_expr, _ = truncated_fft(func, order=3, point=2.51, N=50, ds=0.1, plot=True)
    fourier_func = sympy.lambdify(x, fourier_expr, 'jax')

    # Taylor series approximation
    taylor_expr = taylor_approx(func, point=2.51, order=4, plot=True)
    taylor_func = sympy.lambdify(x, taylor_expr, 'jax')

    # Analyze the function and its approximations
    analyze_function(func, func_expr)
    analyze_function(taylor_func, taylor_expr)
    # For Fourier series, manually find the roots if necessary
    analyze_function(fourier_func, fourier_expr, real_roots=[])
'''
