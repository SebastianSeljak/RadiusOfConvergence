from enum import nonmember

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from scipy.optimize import root_scalar, fsolve
import sympy


x = sympy.Symbol('x')


def func1(x):
   return jnp.square(x) - 6
func1_expr = x**2 - 6
func1_solutions = sympy.solve(func1_expr, x)
func1_solutions = [sol.evalf() for sol in func1_solutions if sol.is_real]
func1_solutions.sort()


def func2(x):
   return jnp.square(x) - (2*x) - 4
func2_expr = x**2 - 2*x - 4
func2_solutions = sympy.solve(func2_expr, x)
func2_solutions = [sol.evalf() for sol in func2_solutions if sol.is_real]
func2_solutions.sort()

def func3(x):
   return -jnp.square(x) - (2*x)
func3_expr = -x**2 - 2*x
func3_solutions = sympy.solve(func3_expr, x)
func3_solutions = [sol.evalf() for sol in func3_solutions if sol.is_real]
func3_solutions.sort()

def func4(x):
   return jnp.power(x, 3) + 2 * jnp.power(x, 2) + (4*x) + 3
func4_expr = x**3 + 2*x**2 + 4*x + 3
func4_solutions = sympy.solve(func4_expr, x)
func4_solutions = [sol.evalf() for sol in func4_solutions if sol.is_real]
func4_solutions.sort()

def func5(x):
   return -jnp.power(x, 3) - jnp.power(x, 2) +  (5 * x)
func5_expr = -x**3 - x**2 + 5*x
func5_solutions = sympy.solve(func5_expr, x)
func5_solutions = [sol.evalf() for sol in func5_solutions if sol.is_real]
func5_solutions.sort()

def func6(x):
   return jnp.cos(x)
func6_expr = sympy.cos(x)
func6_solutions = sympy.solve(func6_expr, x)
func6_solutions = [sol.evalf() for sol in func6_solutions if sol.is_real]
func6_solutions.sort()

def func7(x):
   return -4 * jnp.sin(x) + jnp.cos(x)
func7_expr = -4 * sympy.sin(x) + sympy.cos(x)
func7_solutions = sympy.solve(func7_expr, x)
func7_solutions = [sol.evalf() for sol in func7_solutions if sol.is_real]
func7_solutions.sort()

def func8(x):
   return (2*x) + 1
func8_expr = 2*x + 1
func8_solutions = sympy.solve(func8_expr, x)
func8_solutions = [sol.evalf() for sol in func8_solutions if sol.is_real]
func8_solutions.sort()



functions = [func1, func2, func3, func4, func5, func6, func7, func8]
solutions = [func1_solutions, func2_solutions, func3_solutions, func4_solutions, func5_solutions, func6_solutions, func7_solutions, func8_solutions]

#for i in range(len(functions)):
  # for sol in solutions[i]:
     # print(functions[i](float(sol)))


def newtons_method(func, x0, tol=1e-6, max_iter=100):
   x = x0
   for i in range(max_iter):
       x = x - func(x) / grad(func)(x)
       if abs(func(x)) < tol:
           return x
   return x



def converges2(func, solutions, x0, i):# this checks the convergence given a specific root
   return np.isclose(float(newtons_method(func, x0)), float(solutions[i]))


def radius_of_convergence_bisection2(func, solutions, max_x, root_index, max_iter=100):
   min_radius = float(solutions[root_index])
   if len(solutions) > (root_index + 1):#if there are more roots, set the max radius to the next root
      max_radius = float(solutions[root_index + 1])
   else:
      max_radius = max_x

   root = solutions[root_index]#the root we are looking at
   maxRadius_convergence = converges2(func, solutions, max_radius, root_index)#the convergence of the maxRadius
   minRadius_convergence = converges2(func, solutions, min_radius, root_index)#the convergence of the minRadius

   if maxRadius_convergence:#if the max radius converges then we can conclude that the max radius is infinty
      return None
   if minRadius_convergence == False:
      return 0.0

   midpoint = float((root + max_radius) / 2)
   radius = 0.0
   for i in range(max_iter):#bisection method for finding the radius of convergence
      midpoint_convergence = converges2(func, solutions, midpoint, root_index)
      if i == max_iter - 1:#keep this
         if midpoint_convergence == True:
            break
      if midpoint_convergence == True:
         min_radius = midpoint
         midpoint = float((midpoint + max_radius) / 2)
      if midpoint_convergence == False:
         max_radius = midpoint
         midpoint = float((midpoint + min_radius) / 2)

      if abs(float(midpoint - root)) <= 1e-6 or abs(float(midpoint - max_radius)) <= 1e-6:
         break


   return abs(midpoint-root)









#print(radius_of_convergence_bisection2(func5, func5_solutions, 2, 1))
#print(radius_of_convergence_bisection2(func6, func6_solutions, 0.0, 0))
print(converges2(func6, func6_solutions, 2.74, 0))

