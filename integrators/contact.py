import numpy as np

from integrators.common import getsteps


def contact(init, tspan, h, a, acc, forcing):
    """
    Integrate the damped oscillator with damping factor a
    using the first order contact variational integrator.
    """
    steps = getsteps(tspan, h)
    hsq = np.math.pow(h, 2)
    t0, _ = tspan

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        xnew = x + (h-hsq*a)*p - 0.5*hsq*acc(x) + 0.5*hsq*forcing(t0+h*i)
        pnew = (1.0-h*a)*p + 0.5*h*(
            forcing(t0+h*i) + forcing(t0+h*(i+1)) - acc(x) - acc(xnew)
        )
        sol[i+1] = np.array((pnew, xnew))
    return sol


# Note: this is no longer discussed in the paper but is a
#       straightforward modification of the arguments presented there.
def midpoint(init, tspan, h, a):
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
        xnew = (h - hsq*a)/(1.0 + 0.25*hsq) * p \
            + (1.0-0.25*hsq)/(1.0+0.25*hsq)*x
        pnew = (xnew-x)/h - 0.25*h*(x+xnew)
        sol[i+1] = np.array((pnew, xnew))
    return sol


def symcontact(init, tspan, h, a, acc, forcing):
    """
    Integrate the damped oscillator with damping factor a
    using the second order contact variational integrator.
    """
    steps = getsteps(tspan, h)
    hsq = np.math.pow(h, 2)
    t0, _ = tspan

    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        xnew = x + (h - 0.5*hsq*a)*p - 0.5*hsq*acc(x) + 0.5*hsq*forcing(t0+h*i)
        pnew = (1.0-0.5*h*a)/(1.0 + 0.5*h*a)*p + 0.5*h*(
            forcing(t0+h*i) + forcing(t0+h*(i+1)) - acc(x) - acc(xnew)
        )/(1.0 + 0.5*h*a)
        sol[i+1] = np.array((pnew, xnew))
    return sol
