#! usr/bin/env python 

import sys
sys.path.append('/Users/oliversmith/Desktop/iso_octo/SHTOOLS-3.1')
import pyshtools as sh
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import quadrature_rules as qr 
from itertools import groupby

#Colours
Icolour = '#DB2420' #Central line red
Ocolour = '#00A0E2' #Victoria line blue
O2colour = '#868F98' #Jubilee line grey
D2colour = '#F386A0' #Hammersmith line pink
D4colour = '#97015E' #Metropolitan line magenta
D6colour = '#B05F0F' #Bakerloo line brown
D5colour = '#00843D' #District line green
O3colour = '#021EA9' 

def plotchart(u, mat=False, moredetail=True, cmap='viridis', decoration=True, threeD=False):

    if mat==False:
        #Load in state from file
        n = int(np.sqrt(len(u)/2))
        x = u.reshape((n,2*n))

        #Use Spherical harmonics to interpolate onto a finer grid
        # if desired
        if moredetail==True:
            band_lim = n/2 - 1
            cilm = sh.SHExpandDH(x, sampling=2)
            u1 = sh.MakeGrid2D(cilm, band_lim, interval=1)
        else:
            u1 = x
    else:

        #Use Spherical harmonics to interpolate onto a finer grid
        # if desired
        if moredetail==True:
            n = np.shape(u)[0]
            band_lim = n/2 - 1
            cilm = sh.SHExpandDH(u, sampling=2)
            u1 = sh.MakeGrid2D(cilm, band_lim, interval=1)
        else:
            u1 = u

    #Set norm
    norm=colors.Normalize(vmin = np.min(u1),
                          vmax = np.max(u1), clip=False)

    #Plot chart
    if threeD:
    	#Make 3D lat-lon grid
	    N = np.shape(u1)[0]
	    M = np.shape(u1)[1]
	    lat = np.linspace(-3.*np.pi/2., np.pi/2., M)
	    lon = np.linspace(0.0, np.pi, N)

	    #Make sphere's surface
	    lat, lon = np.meshgrid(lat, lon)
	    r = 1.0
	    x = r*np.sin(lon)*np.cos(lat)
	    y = r*np.sin(lon)*np.sin(lat)
	    z = r*np.cos(lon)

	    #Plot the surface
	    fig = plt.figure()
	    ax = fig.add_subplot(111, projection='3d')
	    surf1 = ax.plot_surface(x,y,z,
	                            rstride=1,cstride=1,cmap='viridis',
	                            facecolors=cm.viridis(norm(u1)))

	    #Set colorbar
	    if decoration==True:
	        m = cm.ScalarMappable(cmap='viridis', norm=norm)
	        m.set_array(u1)
	        cbar = plt.colorbar(m, shrink=0.5)

	    #Display
	    ax.set_aspect("equal")
	    ax.set_xlim([-1,1])
	    ax.set_ylim([-1,1])
	    ax.set_zlim([-1,1])
	    plt.tight_layout()
	    if decoration==False:
	        plt.axis("off")
	    plt.show()

    else:
	    plt.imshow(u1, cmap='viridis', norm=norm)
	    if decoration==True:
	        plt.colorbar(shrink=0.5)
	    else:
	        plt.axis("off")
	    plt.xlabel('$\\theta$',fontsize=26)
	    plt.ylabel('$\\phi$', fontsize=26, rotation=0, labelpad=15)
	    plt.xticks([])
	    plt.yticks([])
	    plt.tight_layout()
	    plt.show()


def harmonicplot(u, ynorm=None, threeD=False, detail=2, cmap='virdis', decoration=True):

    #Load in appropriate grid
    l = len(u)
    if l == 974:
        theta, phi, w = qr.gen_grid()
    else:
        theta, phi, w = qr.generate_iso_grid('model/quadraturedata/qsph1-100-3432DP.txt')

    #Shift grid to SH covention
    phid = np.degrees(phi)
    thetad = np.degrees(theta)
    thetad += 180.0
    phid -= 90.0

    #Do spherical harmonic transform
    cilm = sh.SHExpandLSQ(u, phid, thetad, 20)[0] 
    if ynorm != None:
        cilm_norm = sh.SHExpandLSQ(ynorm, phid, thetad, 20)[0]

    #Do inverse transform onto lat-lon grid
    raster = sh.MakeGrid2D(cilm, 20, interval=detail)
    if ynorm != None:
        raster_norm = sh.MakeGrid2D(cilm_norm, 20, interval=detail)

    #Set colourmap
    if ynorm == None: 
        norm = colors.Normalize(vmin = np.min(raster), vmax = np.max(raster),
                            clip = False)
    else:
        norm = colors.Normalize(vmin = np.min(raster_norm), vmax = np.max(raster_norm-0.2),
                            clip = False)

    #Plot image
    if threeD == False:
        plt.imshow(raster, cmap='viridis', norm=norm)
        if decoration==True:
            plt.colorbar(shrink=0.5)
        plt.xlabel('$\\theta$',fontsize=26)
        plt.ylabel('$\\phi$', fontsize=26, rotation=0, labelpad=15)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        if decoration==False:
            plt.axis("off")
        plt.show()
    else:
        #Make lat-long grid
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf1 = ax.plot_surface(x,y,z,
            rstride=1,cstride=1, cmap='viridis',
            facecolors=cm.viridis(norm(raster)))

        #Display
        ax.set_aspect("equal")
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        if decoration==True:
            m = cm.ScalarMappable(cmap='viridis', norm=norm)
            m.set_array(raster)
            cbar = plt.colorbar(m, shrink=0.5)
        else:
            plt.axis("off")
        plt.tight_layout()
        plt.show()


def plot_bif(p, u, s):

    #Group according to stability
    plt.ion() 
    for g_s, group in groupby(zip(zip(p, u), s), lambda p: p[1]):
        g_p, g_u = [], []
        for i in group:
            g_p.append(i[0][0])
            g_u.append(i[0][1])
        #Plot as dashed or solid line
        if g_s:
            plt.plot(g_p, g_u, linestyle='--', color=Ocolour, linewidth=2.0)
        else:
            plt.plot(g_p, g_u, linestyle='-', color=Ocolour, linewidth=3.0)

    plt.ylabel('$||u||_{2}$', fontsize=24, rotation=0, labelpad = 26)
    plt.xlabel('$h$', fontsize=24)
    plt.tight_layout()
    plt.pause(0.001) 


if __name__=="__main__":
	main() 









































