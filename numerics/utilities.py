#! usr/bin/env python

import numpy as np

def full_jacobian(dfunc, u):

	n = len(u)
	mat = np.zeros((n,n)) 
	e = np.zeros(n)

	for i in range(n):
		e[i] = 1.0
		mat[i,:] = dfunc(e, u)
		e[i] = 0.0 

	return mat 