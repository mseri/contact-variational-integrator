import numpy as np

from integrators.common import getsteps


def contact(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a
    using the first order contact variational integrator.
    """
    steps = getsteps(tspan, h)
    hsq = np.math.pow(h, 2)

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        xnew = (1.0-hsq/2.0)*x + (h-hsq*a)*p
        pnew = (1.0-h*a)*p - h/2.0*(xnew + x)
        sol[i+1] = np.array((pnew, xnew))
    return sol


# Note: this is no longer discussed in the paper but is a
#       straightforward modification of the arguments presented there.
def midpoint(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a
    using the first order midpoint contact variational integrator.
    """
    steps = getsteps(tspan, h)
    hsq = np.math.pow(h, 2)

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        xnew = (h - hsq*a)/(1.0 + hsq/4.0) * p + (1.0-hsq/4.0)/(1.0+hsq/4.0)*x
        pnew = (xnew-x)/h - h/4.0*(x+xnew)
        sol[i+1] = np.array((pnew, xnew))
    return sol


def symcontact(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a
    using the second order contact variational integrator.
    """
    steps = getsteps(tspan, h)
    hsq = np.math.pow(h, 2)

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        xnew = (1.0-hsq/2.0)*x + (h - hsq*a/2.0)*p
        pnew = (1.0-h*a/2.0)/(1.0+h*a/2.0)*p - h/2.0*(xnew + x)/(1.0+h*a/2.0)
        sol[i+1] = np.array((pnew, xnew))
    return sol
