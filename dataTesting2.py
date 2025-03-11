import numpy as np
import jax.numpy as jnp
import jax
from jax import grad
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
    return np.round(float(newtons_method(func, x0)), 6) == np.round(float(solution), 6)


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
        approx += derivative_at_point / sympy.factorial(i) * (x - point) ** i
    return approx

def taylor_approx_jnp(func, point=0.0, order=3):
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
        approx += derivative_at_point / jnp.factorial(i) * jnp.power(x - point, i)
    return approx


def truncated_fft(function, n_terms, N=int(1e6), ds=1e-3, plot=False):
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
    x_np = np.linspace(-T / 2, T / 2, N, endpoint=False)  # NumPy array for FFT
    x_sp = sympy.Symbol('x')
    data = function(x_np)  # Evaluate the function using NumPy
    fft_result = np.fft.fft(data)
    truncated_fft = np.zeros_like(fft_result, dtype=complex)

    truncated_fft[:n_terms] = fft_result[:n_terms]
    truncated_fft[-n_terms:] = fft_result[-n_terms:]

    reconstructed_data = np.fft.ifft(truncated_fft)

    f0 = 1 / T
    a0 = 2 * fft_result[0].real / N
    fourier_series_expr = a0 / 2

    for k in range(1, n_terms + 1):
        ak_complex = 2 * fft_result[k] / N
        ak = ak_complex.real
        bk = -ak_complex.imag
        fourier_series_expr += ak * sympy.cos(2 * sympy.pi * k * f0 * x_sp) + bk * sympy.sin(
            2 * sympy.pi * k * f0 * x_sp)

    if plot:
        plt.plot(x_np, data, label="Original data")
        plt.plot(x_np, reconstructed_data.real, label=f"Reconstructed data (n={n_terms})")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Original vs. Reconstructed Data")
        plt.show()

    return fourier_series_expr, reconstructed_data.real



#Functions

def test_func1(x):
    return jnp.exp(x) - 1
test_func1_expr = sympy.exp(x) - 1
test_func1_fournier = truncated_fft(test_func1, 5)

def func1_approx(x):
    return taylor_approx_jnp(test_func1, 0.0, 4)

func1_approx_expr = taylor_approx(test_func1, 0.0, 4)


def test_func2(x):
    return jnp.exp(2*x) - 2
test_func2_expr = sympy.exp(2*x) - 2
test_func2_fournier = truncated_fft(test_func2, 5)

def func2_approx(x):
    return taylor_approx_jnp(test_func2, 0.0, 4)

func2_approx_expr = taylor_approx(test_func2, 0.0, 4)

def test_func3(x):
    return jnp.exp(3*x) + 4*x
test_func3_expr = sympy.exp(3*x) + 4*x
test_func3_fournier = truncated_fft(test_func3, 5)

def func3_approx(x):
    return taylor_approx_jnp(test_func3, 0.0, 4)

func3_approx_expr = taylor_approx(test_func3, 0.0, 4)

def test_func4(x):
    return jnp.log(x)
test_func4_expr = sympy.log(x)
test_func4_fournier = truncated_fft(test_func4, 5)

def func4_approx(x):
    return taylor_approx_jnp(test_func4, 1.0, 4)

func4_approx_expr = taylor_approx(test_func4, 1.0, 4)

def test_func5(x):
    return jnp.log(x + 1)
test_func5_expr = sympy.log(x + 1, e)
test_func5_fournier = truncated_fft(test_func5, 5)

def func5_approx(x):
    return taylor_approx_jnp(test_func5, 0.0, 4)

func5_approx_expr = taylor_approx(test_func5, 0.0, 4)

def test_func6(x):
    return -jnp.log(x)
test_func6_expr = -sympy.log(x, e)
test_func6_fournier = truncated_fft(test_func6, 5)

def func6_approx(x):
    return taylor_approx_jnp(test_func6, 1.0, 4)

func6_approx_expr = taylor_approx(test_func6, 1.0, 4)

def test_func7(x):
    return jnp.log(x) + 3*x
test_func7_expr = sympy.log(x, e) + 3*x
test_func7_fournier = truncated_fft(test_func7, 5)

def func7_approx(x):
    return taylor_approx_jnp(test_func7, 1.0, 4)

func7_approx_expr = taylor_approx(test_func7, 1.0, 4)

'''
def test_func8(x):
    return jnp.log(x) + jnp.exp(x)
test_func8_expr = sympy.log(x, e) + sympy.exp(x)
'''

functions = {test_func1, test_func2, test_func3, test_func4, test_func5, test_func6, test_func7}
function_expressions = {test_func1_expr, test_func2_expr, test_func3_expr, test_func4_expr, test_func5_expr, test_func6_expr, test_func7_expr}
#fournier_approx = [test_func1_fournier, test_func2_fournier, test_func3_fournier, test_func4_fournier, test_func5_fournier, test_func6_fournier, test_func7_fournier]
taylor_func = {func1_approx, func2_approx, func3_approx, func4_approx, func5_approx, func6_approx, func7_approx}
taylor_func_expression = {func1_approx_expr, func2_approx_expr, func3_approx_expr, func4_approx_expr, func5_approx_expr, func6_approx_expr, func7_approx_expr}


for func, func_expr, taylor, taylor_expr in zip(functions, function_expressions, taylor_func, taylor_func_expression):
    analyze_function(func, func_expr)
    print("Taylor Approximation:")
    analyze_function(taylor, taylor_expr)
