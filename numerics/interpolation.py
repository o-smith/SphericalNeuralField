#! usr/bin/env python 

import sys
# sys.path.append('/usr/lib64/python2.7/site-packages/pyshtools')
sys.path.append('/Users/oliversmith/iso_octo/SHTOOLS-3.1') 
# import _SHTOOLS as sh
import pyshtools as sh
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def interp_norm(u, phi, theta):
    
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
         
    
def harmonicplot(theta, phi, u, un, fname, band_lim=20, threeD=False):

	fig = plt.figure()
	if threeD:
		ax = fig.add_subplot(111, projection="3d")
	else:
		ax = fig.add_subplot(111)
    
	#Convert grid to match the conventions of the SH library
	phid = np.degrees(phi)
	thetad = np.degrees(theta)
	thetad += 180.0
	phid += 90.0

	#Do spherical harmonic transform
	cilm, chi2 = sh.SHExpandLSQ(u, phid, thetad, band_lim)
	cilm_n, chi2_n = sh.SHExpandLSQ(un, phid, thetad, band_lim)

	#Do inverse spherical harmonic transform onto lat-lon grid
	raster = sh.MakeGrid2D(cilm,  band_lim, interval=1)
	raster_n = sh.MakeGrid2D(cilm_n,  band_lim, interval=1)

	#Set colour map
	norm=colors.Normalize(vmin = np.min(raster_n),
	                      vmax = np.max(raster_n), clip = False)

	m = cm.ScalarMappable(cmap="viridis", norm=norm)
	m.set_array(u)
	                      

	#Render image as either a chart or a 3D image
	if threeD==False:
		ax.imshow(raster, cmap='viridis', norm=norm)
		plt.colorbar(m, shrink=0.5)
		ax.set_xlabel('$\\theta$',fontsize=26)
		ax.set_ylabel('$\phi$', fontsize=26, rotation=0, labelpad=15)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.tight_layout()

	else:
		#Make 3D lat-lon grid
		N = np.shape(raster)[0]
		M = np.shape(raster)[1]
		lat = np.linspace(0.0, 2.0*np.pi, M)
		lon = np.linspace(0.0, np.pi, N)

		#Make sphere's surface
		lat, lon = np.meshgrid(lat, lon)
		x = np.sin(lon)*np.cos(lat)
		y = np.sin(lon)*np.sin(lat)
		z = np.cos(lon)

		#Plot the surface
		surf1 = ax.plot_surface(x,y,z,
		    rstride=1,cstride=1, cmap='viridis', 
		    facecolors=cm.viridis(norm(raster))) 
		    
		m = cm.ScalarMappable(cmap='viridis', norm=norm)
		m.set_array(raster_n)
		ax.set_aspect('equal')
		ax.axis("off")
		plt.colorbar(m, shrink=0.5) 
	    
	if fname == None:
		plt.show()
	else:
		plt.savefig(fname, format='jpg')
		ax.cla()
		plt.close()


if __name__=="__main__":
	main() 

