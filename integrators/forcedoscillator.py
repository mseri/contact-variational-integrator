import numpy as np

import integrators.common as com
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
    f = forcing(beta, omega)
    return sym.euler(init, tspan, h, lambda x, p, t: -x-a*p+f(t))


def leapfrog(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using Leapfrog.
    """
    f = forcing(beta, omega)
    return sym.leapfrog_implicit(init, tspan, h, lambda x, p, t: -x-a*p+f(t))


def leapfrog2(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using single step
    Leapfrog for separable Hamiltonians.
    """
    f = forcing(beta, omega)
    return sym.leapfrog(init, tspan, h, lambda x, p, t: -x-a*p+f(t))


def ruth3(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using Ruth3
    for separable Hamiltonians.
    """
    f = forcing(beta, omega)
    return sym.ruth3(init, tspan, h, lambda x, p, t: -x-a*p+f(t))


def ruth4(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using Ruth4
    for separable Hamiltonians.
    """
    f = forcing(beta, omega)
    return sym.ruth4(init, tspan, h, lambda x, p, t: -x-a*p+f(t))


def pseudoleapfrog(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a using pseudo Leapfrog
    for separable Hamiltonians in the sense of Candy, Rozmus.
    """
    f = forcing(beta, omega)
    return sym.pseudoleapfrog(init, tspan, h, lambda x, p, t: -x-a*p+f(t))


# Contact integrators for system with forcing

def contact(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a
    using the first order contact variational integrator.
    """
    return con.contact(init, tspan, h, a, lambda x: x, forcing(beta, omega))


def symcontact(init, tspan, a, beta, omega, h):
    """
    Integrate the damped oscillator with damping factor a
    using the second order contact variational integrator.
    """
    return con.symcontact(init, tspan, h, a, lambda x: x, forcing(beta, omega))

def variational_noncontact(init, tspan, a, b, omega, h):
    """
    Integrate the damped oscillator with damping factor a
    using the second order variational integrator from
    D Martin de Diego and R Sato Martin de Almagro.
    Variational order for forced Lagrangian systems.
    Nonlinearity, Volume 31, Number 8 (2018).
    """
    return com.variational_noncontact(init, tspan, h, a, b, omega)
