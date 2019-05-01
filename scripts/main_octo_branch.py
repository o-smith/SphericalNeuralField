#! usr/bin/env python 

import numpy as np 
from model.fields import * 
from numerics.continuation import * 
from numerics.interpolation import * 

#Make the neural field model 
field = SphericalQuadratureNeuralField() 
field.makeGrid("lebedev") 
field.computeKernel() 

#Create a flat state, and set the main bifurcation parameter,
#in this case h, to some high value  
u = np.zeros(len(field.phi)) 
p = np.zeros(8) 
p[0] = 3.0 
p[1] = 8.0
p[2] = 49.3155529412
p[3] = 1.0
p[4] = 6.6
p[5] = 1.0/28.0
p[6] = 5.0
p[7] = 1.0/20.0
field.param_unpack(p)

measure = lambda u: interp_measure(u, field.phi, field.theta)

#Now bisect onto the point where the ground bifurcation occurs 
xi1, xi0, p = bisect(field.makeJv, field.makeF, measure, u, p, target=0.35)
print p[0] 
