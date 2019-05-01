#! usr/bin/env python 

import numpy as np 
from model.fields import * 
from numerics.continuation import * 
from numerics.interpolation import * 

#Make the neural field model 
field = SphericalQuadratureNeuralField() 
field.makeGrid("icosahedral")
field.computeKernel() 

#Load in a state from the octahedral branch and set the 
#system parameters to match this point on the branch 
u = np.genfromtxt("data/icos_states/state_0.621556_78.590470.txt") 
p = np.zeros(8) 
p[0] = 0.621556 
p[1] = 8.0
p[2] = 49.3155529412
p[3] = 1.0
p[4] = 6.6
p[5] = 1.0/28.0
p[6] = 5.0
p[7] = 1.0/20.0
field.param_unpack(p)

measure = lambda u: interp_measure(u, field.phi, field.theta) 

secant_continuation(field.makeJv, field.makeF, measure, u, p, txtfilename="test.txt")