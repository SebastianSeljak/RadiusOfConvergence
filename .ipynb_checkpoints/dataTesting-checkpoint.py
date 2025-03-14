import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, scipy
import sympy

x = sympy.Symbol('x')
e = jnp.e

def newtons_method(func, x0, tol=1e-6, max_iter=100):
   x = x0
   for i in range(max_iter):
       x = x - func(x) / grad(func)(x)
       if jnp.abs(func(x)) < tol:  # changed to using jax absolute value to allow for compatibility with other functions
           return x
   return x


def converges(func, solution, x0):# this checks the convergence given a specific root
  return np.isclose(float(newtons_method(func, x0)), float(solution))


def radius_of_convergence_bisection(func, solutions, max_x=100., max_iter=100):
    radii_of_convergence = []
    for i, solution in enumerate(solutions):
        # Check positive side first:
        # Set min and max bounds for the radius of convergence
        min_radius = float(solution)
        if len(solutions) > (i + 1): #if there are more roots, set the max radius to the next root
            max_radius = float(solutions[i + 1])
        else:
            max_radius = max_x

        # Check if bounds converge
        maxRadius_convergence = converges(func, solution, max_radius) # Boolean status of max convergence
        minRadius_convergence = converges(func, solution, min_radius) # Boolean status of min convergence

        if maxRadius_convergence: # Should have additional checks here
            upper_radius = None
        elif not minRadius_convergence:
            upper_radius = 0.0
        else:
            midpoint = float((solution + max_radius) / 2)
            for j in range(max_iter):  # bisection method for finding the radius of convergence
                midpoint_convergence = converges(func, solution, midpoint)
                if midpoint_convergence:
                    min_radius = midpoint
                    midpoint = ((midpoint + max_radius) / 2)
                else:
                    max_radius = midpoint
                    midpoint = ((midpoint + min_radius) / 2)

                if abs((midpoint - solution)) <= 1e-6 or abs((midpoint - max_radius)) <= 1e-6:
                    break

            upper_radius = np.round(midpoint, 5)

        # Perform the same procedure below the min radius
        min_radius = float(solution)
        max_radius = float(solutions[i - 1]) if i > 0 else -max_x


        maxRadius_convergence = converges(func, solution, max_radius)#the convergence of the maxRadius
        minRadius_convergence = converges(func, solution, min_radius)#the convergence of the minRadius

        if maxRadius_convergence:#if the max radius converges then we can conclude that the max radius is infinty
            lower_radius = None
        elif not minRadius_convergence:
            lower_radius = 0.0
        else:
            midpoint = float((solution + max_radius) / 2)
            for i in range(max_iter):#bisection method for finding the radius of convergence
                midpoint_convergence = converges(func, solution, midpoint)
                if i == max_iter - 1:#keep this
                        if midpoint_convergence == True:
                            break
                if midpoint_convergence == True:
                        min_radius = midpoint
                        midpoint = float((midpoint + max_radius) / 2)
                if midpoint_convergence == False:
                        max_radius = midpoint
                        midpoint = float((midpoint + min_radius) / 2)

                if abs(float(midpoint - solution)) <= 1e-6 or abs(float(midpoint - max_radius)) <= 1e-6:
                        break

            lower_radius = np.round(midpoint,5)
        radii_of_convergence.append((lower_radius, upper_radius))
    return radii_of_convergence


def deriv_of_order(func, order):
    output = func
    for i in range(order):
        output = grad(output)
    return output

def taylor_approx(func, point=0.0, order=3):
    approx = 0
    for i in range(order + 1):
        approx += deriv_of_order(func, i)(point) / scipy.special.factorial(i, exact=False) * (x - point)**i
    return approx

def taylor_approx_jnp(func, point=0.0, order=3):
    approx = 0
    for i in range(order + 1):
        approx += deriv_of_order(func, i)(point) / scipy.special.factorial(i, exact=False) * jnp.power(x - point, i)
    return approx


#Functions

def test_func1(x):
    return jnp.exp(x) - 1
test_func1_expr = sympy.exp(x) - 1
test_func1_solutions = sympy.solve(test_func1_expr, x)
test_func1_solutions = [sol.evalf() for sol in test_func1_solutions if sol.is_real]
test_func1_solutions.sort()

def func1_approx(x):
    return taylor_approx_jnp(test_func1, 0.0, 4)
func1_approx_expr = taylor_approx(test_func1, 0.0, 4)
test_func1approx_solutions = sympy.solve(func1_approx_expr, x)
test_func1approx_solutions = [sol.evalf() for sol in test_func1approx_solutions if sol.is_real]
test_func1approx_solutions.sort()

def test_func2(x):
    return jnp.exp(2*x) - 2
test_func2_expr = sympy.exp(2*x) - 2
test_func2_solutions = sympy.solve(test_func2_expr, x)
test_func2_solutions = [sol.evalf() for sol in test_func2_solutions if sol.is_real]
test_func2_solutions.sort()

def func2_approx(x):
    return taylor_approx_jnp(test_func2, 0.0, 4)
func2_approx_expr = taylor_approx(test_func2, 0.0, 4)
test_func2approx_solutions = sympy.solve(func2_approx_expr, x)
test_func2approx_solutions = [sol.evalf() for sol in test_func2approx_solutions if sol.is_real]
test_func2approx_solutions.sort()


def test_func3(x):
    return jnp.exp(3*x) + 4*x
test_func3_expr = sympy.exp(3*x) + 4*x
test_func3_solutions = sympy.solve(test_func3_expr, x)
test_func3_solutions = [sol.evalf() for sol in test_func3_solutions if sol.is_real]
test_func3_solutions.sort()

def func3_approx(x):
    return taylor_approx_jnp(test_func3, 0.0, 4)
func3_approx_expr = taylor_approx(test_func3, 0.0, 4)
test_func3approx_solutions = sympy.solve(func3_approx_expr, x)
test_func3approx_solutions = [sol.evalf() for sol in test_func3approx_solutions if sol.is_real]
test_func3approx_solutions.sort()


def test_func4(x):
    return jnp.log(x)
test_func4_expr = sympy.log(x)
test_func4_solutions = sympy.solve(test_func4_expr, x)
test_func4_solutions = [sol.evalf() for sol in test_func4_solutions if sol.is_real]
test_func4_solutions.sort()

def func4_approx(x):
    return taylor_approx_jnp(test_func4, 1.0, 4)
func4_approx_expr = taylor_approx(test_func4, 1.0, 4)
test_func4approx_solutions = sympy.solve(func4_approx_expr, x)
test_func4approx_solutions = [sol.evalf() for sol in test_func4approx_solutions if sol.is_real]
test_func4approx_solutions.sort()


def test_func5(x):
    return jnp.log(x + 1)
test_func5_expr = sympy.log(x + 1, e)
test_func5_solutions = sympy.solve(test_func5_expr, x)
test_func5_solutions = [sol.evalf() for sol in test_func5_solutions if sol.is_real]
test_func5_solutions.sort()

def func5_approx(x):
    return taylor_approx_jnp(test_func5, 0.0, 4)
func5_approx_expr = taylor_approx(test_func5, 0.0, 4)
test_func5approx_solutions = sympy.solve(func5_approx_expr, x)
test_func5approx_solutions = [sol.evalf() for sol in test_func5approx_solutions if sol.is_real]
test_func5approx_solutions.sort()


def test_func6(x):
    return -jnp.log(x)
test_func6_expr = -sympy.log(x, e)
test_func6_solutions = sympy.solve(test_func6_expr, x)
test_func6_solutions = [sol.evalf() for sol in test_func6_solutions if sol.is_real]
test_func6_solutions.sort()

def func6_approx(x):
    return taylor_approx_jnp(test_func6, 1.0, 4)
func6_approx_expr = taylor_approx(test_func6, 1.0, 4)
test_func6approx_solutions = sympy.solve(func6_approx_expr, x)
test_func6approx_solutions = [sol.evalf() for sol in test_func6approx_solutions if sol.is_real]
test_func6approx_solutions.sort()


def test_func7(x):
    return jnp.log(x) + 3*x
test_func7_expr = sympy.log(x, e) + 3*x
test_func7_solutions = sympy.solve(test_func7_expr, x)
test_func7_solutions = [sol.evalf() for sol in test_func7_solutions if sol.is_real]
test_func7_solutions.sort()

def func7_approx(x):
    return taylor_approx_jnp(test_func7, 1.0, 4)
func7_approx_expr = taylor_approx(test_func7, 1.0, 4)
test_func7approx_solutions = sympy.solve(func7_approx_expr, x)
test_func7approx_solutions = [sol.evalf() for sol in test_func7approx_solutions if sol.is_real]
test_func7approx_solutions.sort()

'''
def test_func8(x):
    return jnp.log(x, e) + jnp.exp(x)
test_func8_expr = sympy.log(x, e) + sympy.exp(x)
test_func8_solutions = sympy.solve(test_func8_expr, x)
test_func8_solutions = [sol.evalf() for sol in test_func8_solutions if sol.is_real]
test_func8_solutions.sort()
'''

functions = [test_func1, test_func2, test_func3, test_func4, test_func5, test_func6, test_func7]
solutions = [test_func1_solutions, test_func2_solutions, test_func3_solutions, test_func4_solutions, test_func5_solutions, test_func6_solutions, test_func7_solutions]

func_approx = [func1_approx, func2_approx, func3_approx, func4_approx, func5_approx, func6_approx, func7_approx]
approx_solutions = [test_func1approx_solutions, test_func2approx_solutions, test_func3approx_solutions, test_func4approx_solutions, test_func5approx_solutions, test_func6approx_solutions, test_func7approx_solutions]


for approx, sol in zip(func_approx, approx_solutions):
    print(f" radius of convergence for {approx}:")
    print(radius_of_convergence_bisection(approx, sol))


i = 1
for func, solutions in zip(functions, solutions):
    print(f"Radius of Convergence for Function {i} with roots {solutions}")
    print(radius_of_convergence_bisection(func, solutions))
    i = i + 1
