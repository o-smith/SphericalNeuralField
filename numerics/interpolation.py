#! usr/bin/env python 

import sys
# sys.path.append('/usr/lib64/python2.7/site-packages/pyshtools')
sys.path.append('/Users/oliversmith/Desktop/iso_octo/SHTOOLS-3.1')
# import _SHTOOLS as sh
import pyshtools as sh
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy.linalg import norm


def interp_measure(u, phi, theta):
    
    lmax = 40
    n = 2*lmax + 2
    latspacing = 180.0/float(n) 
    
    phid = np.degrees(phi)
    thetad = np.degrees(theta)
    thetad += 180.0
    phid += 90.0
    
    cilm, chi2 = sh.SHExpandLSQ(u, phid, thetad, lmax)
    raster = sh.MakeGrid2D(cilm, lmax, interval=latspacing)
    
    mu = np.mean(raster)
    y = raster - mu
    return norm(y, ord=2)
        


if __name__=="__main__":
	main() 

