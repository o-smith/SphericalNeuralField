#! usr/bin/env python 

import numpy as np 
from model.fields import * 
from numerics.continuation import * 
from numerics.interpolation import * 

#Make the neural field model 
field = SphericalHarmonicNeuralField() 
field.makeGrid(lmax=20) 
field.makeWn() 

#Load in a state from the octahedral branch and set the 
#system parameters to match this point on the branch 
u = np.genfromtxt("data/O2_states/state_0.189430_176.120282.txt") 
p = np.zeros(8) 
p[0] = 0.189430 
p[1] = 8.0
p[2] = 49.3155529412
p[3] = 1.0
p[4] = 6.6
p[5] = 1.0/28.0
p[6] = 5.0
p[7] = 1.0/20.0
field.param_unpack(p)

secant_continuation(field.makeJv, field.makeF, field.finer_measure, u, p, "dump.txt",
			txtfilename="test.txt")