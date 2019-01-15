import numpy as np

import integrators.contact as con
import integrators.symplectic as sym


def forcing(beta, omega):
    """
    Forcing of the form beta*sin(omega*t)
    """
    return lambda t: beta*np.sin(omega*t)


def reference(a, beta, omega):
    """
    Reference solution for damped oscilltor with sinusoidal forcing
    """
    phi = np.arctan2(a*omega, np.power(omega, 2)-1.0)
    prefactor = - beta / \
        np.sqrt(np.power(omega*a, 2) + np.power(1.0-np.power(omega, 2), 2))

    return lambda t: prefactor*np.sin(omega*t + phi)

def dreference(a, beta, omega):
    """
    Reference solution for damped oscilltor with sinusoidal forcing
    """
    phi = np.arctan2(a*omega, np.power(omega, 2)-1.0)
    prefactor = - beta / \
        np.sqrt(np.power(omega*a, 2) + np.power(1.0-np.power(omega, 2), 2))

    return lambda t: prefactor*omega*np.cos(omega*t + phi)

# Symplectic integrators


def euler(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using Symplectic Euler.
    """
    return sym.euler(init, tspan, h, lambda x, p, t: -x-a*p)


def leapfrog(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using Leapfrog.
    """
    return sym.leapfrog(init, tspan, h, lambda x, p, t: -x-a*p)


def ruth3(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using Ruth3.
    """
    return sym.ruth3(init, tspan, h, lambda x, p, t: -x-a*p)


def ruth4(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using Ruth4.
    """
    return sym.ruth4(init, tspan, h, lambda x, p, t: -x-a*p)


def leapfrog2(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using Leapfrog.
    """
    return sym.leapfrog2(init, tspan, h, lambda x, p, t: -x-a*p)


def pseudoleapfrog(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using pseudo Leapfrog
    in the sense of Candy, Rozmus.
    """
    return sym.pseudoleapfrog(init, tspan, h, lambda x, p, t: -x-a*p)


# Contact integrators

def contact(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a
    using the first order contact variational integrator.
    """
    return con.contact(init, tspan, h, a, forcing(beta, omega))


def symcontact(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a
    using the second order contact variational integrator.
    """
    return con.symcontact(init, tspan, h, a, forcing(beta, omega))
