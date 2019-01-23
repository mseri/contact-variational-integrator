import numpy as np
import scipy.optimize as so

from integrators.common import getsteps


def euler(init, tspan, h, acc):
    """
    Symplectic Euler integrator
    """
    steps = getsteps(tspan, h)
    t0, _ = tspan

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        pnew = p + h*acc(x, p, t0+i*h)
        xnew = x + h*pnew
        sol[i+1] = np.array((pnew, xnew))

    return sol


def leapfrog(init, tspan, h, acc):
    """
    Leapfrog integrator (for separable Hamiltonians).
    Defaults to the damped oscillator with damping factor a
    """
    steps = getsteps(tspan, h)
    hsq = np.math.pow(h, 2)
    t0, _ = tspan

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        xnew = x + h*p + hsq/2.0*acc(x, p, t0+i*h)
        pnew = p + h*(acc(x, p, t0+i*h)+acc(xnew, p, t0+(i+1)*h))/2.0
        sol[i+1] = np.array((pnew, xnew))

    return sol


def leapfrog_implicit(init, tspan, h, acc):
    """
    Leapfrog integrator, general implicit form.
    Defaults to the damped oscillator with damping factor a
    """
    steps = getsteps(tspan, h)
    t0, _ = tspan

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        pint = so.fsolve(
            lambda pint: p - pint + 0.5*h*acc(x, pint, t0+i*h),
            p
        )[0]
        xnew = x + h*pint
        pnew = pint + 0.5*h*acc(xnew, pint, t0+(i+1)*h)
        sol[i+1] = np.array((pnew, xnew))

    return sol


# For the values used in the implementations below
# see Candy-Rozmus (https://www.sciencedirect.com/science/article/pii/002199919190299Z)
# and https://en.wikipedia.org/wiki/Symplectic_integrator
def symint_step(init, acc, h, coeffs):
    """
    General symplectic integrator step with coefficients coeffs.
    The parameter init contains the current p,q,t; acc is the acceleration; h is the step.
    """
    p, q, t = init
    for ai, bi in coeffs.T:
        p += bi * acc(q, p, t) * h
        q += ai * p * h
        t += ai * h
    return p, q, t


def symint(init, tspan, h, coeffs, acc):
    """
    General symplectic integrator with coefficients coeffs.
    The parameter init contains the initial values for p and q;
    tspan contains the time span as a tuple; h is the time step;
    acc is the acceleration.
    """
    steps = getsteps(tspan, h)

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    t = tspan[0]
    for i in range(steps-1):
        p0, q0 = sol[i]
        pnew, qnew, t = symint_step((p0, q0, t), acc, h, coeffs)
        sol[i+1] = np.array((pnew, qnew))
    return sol


cleapfrog = np.array([[0.5, 0.5], [0.0, 1.0]])
cpseudoleapfrog = np.array([[1.0, 0.0], [0.5, 0.5]])
cruth3 = np.array([[2.0/3.0, -2.0/3.0, 1.0], [7.0/24.0, 0.75, -1.0/24.0]])

# Note: ruth4 seems to give poor results
# see also the discussion at
# https://scicomp.stackexchange.com/questions/20533/test-of-3rd-order-vs-4th-order-symplectic-integrator-with-strange-result
c = np.math.pow(2.0, 1.0/3.0)
cruth4 = np.array([[0.5, 0.5*(1.0-c), 0.5*(1.0-c), 0.5],
                   [0.0, 1.0, -c, 1.0]]
                  ) / (2.0 - c)


def ruth3(init, tspan, h, acc):
    """
    Integrate using the acceleration acc using Ruth3 for separable Hamiltonians.
    """
    return symint(init, tspan, h, cruth3, acc)


def ruth4(init, tspan, h, acc):
    """
    Integrate using the acceleration acc using Ruth4 for separable Hamiltonians.
    """
    return symint(init, tspan, h, cruth4, acc)


def leapfrog2(init, tspan, h, acc):
    """
    Integrate using the acceleration acc using Leapfrog for separable Hamiltonians.
    """
    return symint(init, tspan, h, cleapfrog, acc)


def pseudoleapfrog(init, tspan, h, acc):
    """
    Integrate using the acceleration acc sing pseudo Leapfrog
    for separable Hamiltonians in the sense of Candy, Rozmus.
    """
    return symint(init, tspan, h, cpseudoleapfrog, acc)
