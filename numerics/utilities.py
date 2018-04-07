#! usr/bin/env python

import sys
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


def update_progress(progress):

    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] ".format( "#"*block + "-"*(barLength-block))
    sys.stdout.write(text)
    sys.stdout.flush()


if __name__=="__main__":
	main() 