#! usr/bin/env python 

from model.fields import *
from numerics.solvers import * 
from model.plotting import harmonicplot
from numerics.utilities import update_progress
from numerics.interpolation import interp_measure    
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.integrate import ode

print "Testing the Neural field convergence using the octahedral quadrature scheme..."

#Make the neural field model 
field = SphericalQuadratureNeuralField() 
field.makeGrid("octahedral")
field.computeKernel() 
field.kappa *= 4.0*np.pi 
field.h = 0.3

#Set up the "function handles"
#These are interfaces to the underlying Fortran routines
problemHandle   = lambda t, u, p:  field.makeF(u,p) #+ fixer

#Initial state 
u1 = np.zeros(field.n)

#Set up RK4 integrator
t0, dt, tmax = 0.0, 0.1, 5.0
du = ode(problemHandle, jac=None)
du.set_integrator('dopri5',rtol=1e-6,nsteps=5000,max_step=0.2)
du.set_initial_value(u1, t0)
pvec = field.param_pack() 
du.set_f_params(pvec)

##Do time-stepping
print "Time stepping..."
while du.t < 100.0:
   # update_progress(du.t/100.0)
   du.integrate(du.t+dt)
   print du.t
u = du.y
print "Time stepped measure = %f" %interp_measure(u, field.phi, field.theta)

#Initialise an object to pass to the solver to 
#record the convergence history 
convergence_recorder = KrylovCounter() 

#Call Newton-GRMES on this state and solve it to make sure it's stationary 
u1, count, info, conv = newton_gmres(field.makeJv, field.makeF, u, pvec, noisy=True, toler=1e-14)
if conv:
	print "Stationary state found in %i iterations." %count 
	print "Measure = %f" %interp_measure(u1, field.phi, field.theta) 
else:
	print "Stationary state not found."
	raise Exception

#Perturb this state with a low amplitude Gaussian bump
print "Perturbing this state..."
uptrb = u1 + field.make_u0(sigma=1.5)*0.01
print "Perturbed measure = %f" %interp_measure(uptrb, field.phi, field.theta) 

#Now pass this perturbed state to Newton-GMRES to see how quickly it converges
#back to the ground state. The convergence recorder object will be passed down
#through Newton's method to GMRES and will capture its convergence during the 
#first iteration of Newton's method 
u2, count, info, conv, err = newton_gmres(field.makeJv, field.makeF, uptrb,pvec, noisy=True,
	toler=1e-14, nmax=20, convobject=convergence_recorder, gmres_tol=1e-15) 

#Print the norm of the newly solved state, it should be the same as before the 
#perturbation was applied
#Print the new norm 
print "Measure of newly found state = %f" %interp_measure(u2, field.phi, field.theta)  

#Plot the convergence results 
fig = plt.figure() 
ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2) 
ax0.semilogy(convergence_recorder.niter, convergence_recorder.resid, 'bo-', ms=5.0)
ax0.grid(True, 'major')
ax0.set_xlabel("Iterations")
ax0.set_ylabel("2-norm residual", rotation=90, labelpad=10) 
ax0.set_title("Inner Krylov convergence", fontsize=14)
ax1.semilogy(err, 'mo-', ms=5.0) 
ax1.grid(True, 'major')
ax1.set_xlabel("Iterations")
ax1.set_ylabel("2-norm residual", rotation=90, labelpad=10) 
ax1.set_title("Convergence of Newton's method")
plt.tight_layout() 
plt.show() 

harmonicplot(u2)






