import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, scipy
import sympy

x = sympy.Symbol('x')
e = jnp.e

from radius_of_convergence import taylor_approx, truncated_fft, analyze_function


# WORKING EXAMPLE
def f4(x):
    return jnp.log(x) # Define a function using jnp
f4_expr = sympy.log(x) # Define its symbolic notation

# First issue: originally, we were evaluating at ln(0), which blows up. Not has been changed to 2.51, and the interval is now 2.51 +/- 50*0.1, so (0.01, 5.01)
f4_fourier_expr, _ = truncated_fft(f4, order=3, point=2.51, N=50, ds=0.1, plot=True) # Your interval is point +/- N*ds. This must be a continuous part of the function
f4_fourier = sympy.lambdify(x, f4_fourier_expr, 'jax') # Convert the symbolic expression to a jax function, not numpy

f4_taylor_expr = taylor_approx(f4, point=2.51, order=4, plot=True) # This is the taylor approximation, also has plotting functionality now
f4_taylor = sympy.lambdify(x, f4_taylor_expr, 'jax') # Same process as above

analyze_function(f4, f4_expr) # This prints out the roots of the function, and correct intervals (I believe, it looks like it's working)
analyze_function(f4_taylor, f4_taylor_expr) # This prints out the roots of the taylor approximation
# HERE WAS SECOND ISSUE: sympy can't solve roots of fourier series. Just copy paste the sympy expression into desmos, set p = 1, i = pi (so that the constants work out),
# and find the roots manually. This is probably the easiest way to do it.
analyze_function(f4_fourier, f4_fourier_expr, real_roots=[-0.1698, 0.95926])


# Quick loops with all the functions, INTERVALS MIGHT BE OFF, i didnt check them
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
