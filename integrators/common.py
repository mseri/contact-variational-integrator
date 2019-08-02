import math
import numpy as np
import matplotlib.pylab as plb


def getsteps(tspan, h):
    """
    Given a timespan and a timestep h, return the number of steps.
    """
    t1, t2 = tspan
    return int(math.floor((t2-t1)/h))


def relerr(ref, sol, shift=10.0):
    """
    Regularized relative error between the reference solution
    and the approximation: abs((shift+sol)/(shift+ref)-1.0)

    As the solutions that we are going to discuss have often a very
    small magnitude, we shift the solutions from 0 to avoid division
    by zero errors. We chose this function to emphasize the relative
    size of the error compared to the solution.
    """
    return np.abs((shift+sol)/(shift+ref)-1.0)


def rk4(init, tspan, a, h):
    """
    RungeKutta4 from matplotlib.pylab for the damped oscillator
    with damping factor a.
    """
    t = np.arange(tspan[0], tspan[1], h)

    def derivs(x, t):
        dp = -x[1] - a*x[0]
        dq = x[0]
        return (dp, dq)
    return plb.rk4(derivs, init, t)


def rk4_forced(init, tspan, a, b, omega, h):
    """
    RungeKutta4 from matplotlib.pylab for the damped oscillator
    with damping factor a and forcing b*sin(omega*t)
    """
    t = np.arange(tspan[0], tspan[1], h)

    def derivs(x, t):
        dp = -x[1] - a*x[0] + b*np.sin(omega*t)
        dq = x[0]
        return (dp, dq)
    return plb.rk4(derivs, init, t)


# second order discretization of a general linearly damped system
# from Verlet discretization, using the construction from
#
# D Martin de Diego and R Sato Martin de Almagro.
# Variational order for forced Lagrangian systems. Nonlinearity, Volume 31, Number 8 (2018)
def variational_noncontact(init, tspan, h, a, b, omega):
    """
    Integrate the damped oscillator with damping factor a
    using the second order variational integrator
    from
    D Martin de Diego and R Sato Martin de Almagro.
    Variational order for forced Lagrangian systems.
    Nonlinearity, Volume 31, Number 8 (2018)
    """
    t0 = tspan[0]
    steps = getsteps(tspan, h)
    hsq = np.math.pow(h, 2)

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        xnew = ((1-hsq/2)*x + h*p) / (1 + h*a/2)
        pnew = p - h*(x+xnew)/2 - a*(xnew - x)/2 + h*b*( np.sin(omega*(t0+h*i)) + np.sin(omega*(t0+h*i+h)) )/2
        sol[i+1] = np.array((pnew, xnew))
    return sol