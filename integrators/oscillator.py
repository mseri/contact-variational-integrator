import integrators.contact as con
import integrators.symplectic as sym


# Symplectic integrators

def euler(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a using Symplectic Euler.
    """
    return sym.euler(init, tspan, h, lambda x, p, t: -x-a*p)


def leapfrog(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a using Leapfrog.
    """
    return sym.leapfrog_implicit(init, tspan, h, lambda x, p, t: -x-a*p)


def leapfrog2(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a using the
    single step Leapfrog integrator for separable Hamiltonians.
    """
    return sym.leapfrog(init, tspan, h, lambda x, p, t: -x-a*p)


def ruth3(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a using Ruth3
    for separable Hamiltonians.
    """
    return sym.ruth3(init, tspan, h, lambda x, p, t: -x-a*p)


def ruth4(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a using Ruth4
    for separable Hamiltonians.
    """
    return sym.ruth4(init, tspan, h, lambda x, p, t: -x-a*p)


def pseudoleapfrog(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a using pseudo Leapfrog
    for separable Hamiltonians in the sense of Candy, Rozmus.
    """
    return sym.pseudoleapfrog(init, tspan, h, lambda x, p, t: -x-a*p)


# Contact integrators

def contact(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a
    using the first order contact variational integrator.
    """
    return con.contact(init, tspan, h, a, lambda x: x, lambda t: 0)


# Note: this is no longer discussed in the paper but is a
#       straightforward modification of the arguments presented there.
def midpoint(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a
    using the first order midpoint contact variational integrator.
    """
    return con.midpoint(init, tspan, h, a)


def symcontact(init, tspan, a, h):
    """
    Integrate the damped oscillator with damping factor a
    using the second order contact variational integrator.
    """
    return con.symcontact(init, tspan, h, a, lambda x: x, lambda t: 0)
