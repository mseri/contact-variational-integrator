# Simulation code for Vermeeren, Seri, Bravetti: Contact Variational Integrators

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mseri/contact-variational-integrator/master)

The notebooks contain the code to generate the pictures used in the [paper](https://arxiv.org/abs/1902.00436),
these can additionally be found in the `img/` folder. The `integators` folder
contains the implementations of all the algorithms tested. The notebooks can
be run remotely via binder, clocking on the badge above, or locally by using
`jupyter`, a recent version of `python`, `numpy` and `matplotlib` is required 
(see `requirements.txt` for a compativle set of dependencies).

The available notebooks are:

- `DampedOscillator.ipynb` and `DampedOscillatorWithForcing.ipynb` contain the code used to generate the pictures in the paper;

- `QuadraticZOscillator.ipynb` contains some explorative code for the oscillator with quadratic z used presented in Example 3. Given that for this case there is no natural comparison algorithm, we decided to omit this from the paper.

