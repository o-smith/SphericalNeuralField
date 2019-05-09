#! usr/bin/env python 

import numpy as np 
from model.fields import * 
from numerics.continuation import * 
from numerics.interpolation import interp_measure
from model.plotting import harmonicplot   

#Make the neural field model 
field = SphericalQuadratureNeuralField() 
field.makeGrid("lebedev")
field.computeKernel() 
field.kappa *= 4.0*np.pi 
field.h = 3.0 
p = field.param_pack() 
print p

#Create a flat state,  
u = np.zeros(field.n) 

print "beginning bisection"
measure = lambda u: interp_measure(u, field.phi, field.theta)
xi1, xi0, p = bisect(field.makeJv, field.makeF, measure, u, p, target=0.35)



# eigvec = np.genfromtxt("data/octo_leb.txt")
# harmonicplot(eigvec)

# #Now bisect onto the point where the ground bifurcation occurs 
# xi1, xi0, p = bisect(field.makeJv, field.makeF, measure, u, p, target=0.35)
# print p[0] 
