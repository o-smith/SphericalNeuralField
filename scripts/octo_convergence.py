#! usr/bin/env python 

from model.fields import *
from numerics.interpolation import * 
from numerics.solvers import *  
from numerics.utilities import * 
from numpy.linalg import cond 
import matplotlib.pyplot as plt 
import numpy as np 
import time 

print "Testing the Neural field convergence using the octahedral Lebedev quadrature scheme..."

#Make the neural field model 
field = SphericalQuadratureNeuralField() 
field.makeGrid("lebedev")
field.computeKernel() 

# plt.imshow(field.kernel, cmap='viridis', interpolation='none')
# plt.colorbar()
# plt.show() 

#Load in a state from the octahedral branch and set the 
#system parameters to match this point on the branch 
u = np.genfromtxt("data/octo_states/state_0.630764_9.142467.txt") 
field.h = 0.630764

#Initialise an object to pass to the solver to 
#record the convergence history 
convergence_recorder = KrylovCounter() 

#Call Newton-GRMES on this state and solve it to make sure it's stationary 
u1, count, info, conv = newtonGMRES(field.makeJv, field.makeF, u, noisy=True, toler=1e-14)
if conv:
	print "Stationary state found on octahedral branch in %i iterations." %count 
else:
	print "Stationary state not found."
	raise Exception

# jac = full_jacobian(field.makeJv, u1+ field.make_u0(sigma=1.5)*0.01 )
# plt.imshow(jac, cmap='viridis', interpolation='none')
# plt.show() 
# print "condition number = %f" %cond(jac)

# harmonicplot(field.theta, field.phi, u1, u1, None)

#Perturb this state with a low amplitude Gaussian bump
print "Perturbing this state..."
uptrb = u1 + field.make_u0(sigma=1.5)*0.04

# # harmonicplot(field.theta, field.phi, uptrb, u1, None)

#Now pass this perturbed state to Newton-GMRES to see how quickly it converges
#back to the ground state. The convergence recorder object will be passed down
#through Newton's method to GMRES and will capture its convergence during the 
#first iteration of Newton's method 
u2, count, info, conv, err = newtonGMRES(field.makeJv, field.makeF, uptrb, noisy=True,
	toler=1e-17, nmax=20, convobject=convergence_recorder, gmres_tol=1e-17) 


#Plot the convergence results 
fig = plt.figure() 
ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2) 
ax0.semilogy(convergence_recorder.niter, convergence_recorder.resid, 'bo-', ms=5.0)
ax0.grid(True, 'major')
ax0.set_xlabel("Iterations")
ax0.set_ylabel("2-norm residual", rotation=90, labelpad=10) 
ax0.set_title("Inner Krylov convergence")
ax1.semilogy(err, 'mo-', ms=5.0) 
ax1.grid(True, 'major')
ax1.set_xlabel("Iterations")
ax1.set_ylabel("2-norm residual", rotation=90, labelpad=10) 
ax1.set_title("Convergence of Newton's method")
plt.show() 
















