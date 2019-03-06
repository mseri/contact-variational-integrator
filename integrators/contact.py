import numpy as np
import sympy as sp
import scipy.optimize as so

from integrators.common import getsteps

def dEL(lag, implicit = True):
    """
    Return the integrator corresponding to a given discrete Lagrangian.
    If implicit = False, a symbolic solution to the EL equations is sought.
    """
    p,x0,x1,z0,z1,h = sp.symbols("p x0 x1 z0 z1 h")
    p0 = sp.expand( - h * sp.diff(lag(x0,x1,z0,z1,h),x0) / ( 1 + h * sp.diff(lag(x0,x1,z0,z1,h),z0) ) )
    p1 = sp.expand( h * sp.diff(lag(x0,x1,z0,z1,h),x1) / ( 1 - h * sp.diff(lag(x0,x1,z0,z1,h),z1) ) )
    
    if implicit:
        xequation = sp.lambdify([p,x0,x1,h], p0 - p) #equation to be solved for x1 at each step
    else:
        pold = sp.symbols("pold")
        [xformula] = sp.solve(p0 - pold,x1) #symbolic expression for next x
        nextx = sp.lambdify([pold,x0,h],xformula)
        
    peq = sp.lambdify([x0,x1,h],p1)
    
    def integrator(init, tspan, stepsize):
        steps = getsteps(tspan, stepsize)
        t0, _ = tspan

        sol = np.empty([steps, 2], dtype=np.float64)
        sol[0] = np.array(init)
        for i in range(steps-1):
            p, x = sol[i]
            if implicit:
                [xnew] = so.fsolve(lambda xnew: xequation(p,x,xnew,stepsize), x + stepsize*p)
            else:
                xnew = nextx(p,x,stepsize)
            pnew = peq(x,xnew,stepsize)
            sol[i+1] = np.array((pnew, xnew))
        return sol
    
    return integrator

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
