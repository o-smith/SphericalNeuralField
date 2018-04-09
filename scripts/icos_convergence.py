#! usr/bin/env python 

from model.fields import *
from numerics.solvers import *   
import matplotlib.pyplot as plt 
import numpy as np 

print "Testing the Neural field convergence using the icosahedral quadrature scheme..."

#Make the neural field model 
field = SphericalQuadratureNeuralField() 
field.makeGrid("icosahedral")
field.computeKernel() 

#Load in a state from the octahedral branch and set the 
#system parameters to match this point on the branch 
u = np.genfromtxt("...") 
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

#Perturb this state with a low amplitude Gaussian bump
print "Perturbing this state..."
uptrb = u1 + field.make_u0(sigma=1.5)*0.04

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