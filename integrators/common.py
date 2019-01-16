import math
import numpy as np
import matplotlib.pylab as plb


def getsteps(tspan, h):
    """
    Given a timespan and a timestep h, return the number of steps.
    """
    t1, t2 = tspan
    return int(math.floor((t2-t1)/h))


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
        dp = -x[1] - a*x[0] - b*np.sin(omega*t)
        dq = x[0]
        return (dp, dq)
    return plb.rk4(derivs, init, t)
