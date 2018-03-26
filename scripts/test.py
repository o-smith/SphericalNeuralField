from model.fields import *
import matplotlib.pyplot as plt 
from scipy.integrate import ode
import numpy as np 

# field = SphericalHarmonicNeuralField() 
# field.makeGrid() 
# field.makeWn() 
# utmp = field.make_u0()
# u0 = utmp.reshape(2*field.n**2)


# makeFt = lambda t, x: field.makeF(x)
# t0, dt = 0.0, 0.1
# du = ode(makeFt, jac=None)
# du.set_integrator('dopri5',rtol=1e-6,nsteps=5000,max_step=0.2)
# du.set_initial_value(u0, t0)


# ##Do time-stepping
# while du.t < 40.0:
#    du.integrate(du.t+dt)
#    print du.t
# u = du.y

# plt.imshow(u.reshape(field.n, 2*field.n), cmap='viridis')
# plt.show() 

field = SphericalQuadratureNeuralField() 
field.makeGrid("icosahedral")
field.computeKernel() 
field.h = 0.0

u0 = field.make_u0()*0.0
makeFt = lambda t, x: field.makeF(x)
t0, dt = 0.0, 0.1
du = ode(makeFt, jac=None)
du.set_integrator('dopri5',rtol=1e-6,nsteps=5000,max_step=0.2)
du.set_initial_value(u0, t0)


##Do time-stepping
while du.t < 100.0:
   du.integrate(du.t+dt)
   print du.t
u = du.y

harmonicplot(field.theta, field.phi, u, u, None)
