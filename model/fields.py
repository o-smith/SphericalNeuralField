import sys
# sys.path.append('/usr/lib64/python2.7/site-packages/pyshtools')
sys.path.append('/Users/oliversmith/iso_octo/SHTOOLS-3.1') 
# import _SHTOOLS as sh
import pyshtools as sh
import numpy as np
import numpy.polynomial.legendre as leg
from scipy.special import legendre
from scipy.linalg import norm, solve


class harmonicNeuralfield:   
    """
    Class for neural fields posed on the 2-sphere.

    Model has a kernel consisting of 
    a difference of Gaussians:
     kernel = a1*exp(-(d^2)/b1) - a2*exp(-(d^2)/b2),
        
    and a sigmoidal firing rate:
        s(u) = 1/(1 + exp(-mu*(u - h))),
        
    and du/dt given by:
        -u + kappa*integral kernel(<x,x'>)s(u(x')) dx',  
        with x in 2-sphere. 
    """ 

    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        #Default attributes
        self.a1, self.b1, self.a2, self.b2 = 6.6, 1.0/28.0, 5.0, 1.0/20.0
        self.h, self.mu, self.kappa = 0.35, 8.0, 50.0
        self.lambda_ord = 10
        self.lmax = 40 
        
        self.n = self.lmax*2 + 2
        latspacing = 180.0/float(self.n)
        lonspacing = 180.0/float(self.n)
        self.lat = np.arange(-90.0, 90.0, latspacing)
        self.lon = np.arange(0.0, 360.0, lonspacing)
        self.phi =  np.deg2rad(self.lat + 90.)
        self.theta = np.deg2rad(self.lon) 
        
                                                
    def kernel(self, xi):
        """Function to compute kernel.
        The argument of this function is 
        xi = cos(theta)=<x,x'>, in [-1,1]."""
        
        d = np.arccos(xi)
        k = self.a1*np.exp(-(d**2)/self.b1) - self.a2*np.exp(-(d**2)/self.b2)
        return k
        
        
    def make_W0(self, quad_ord=60, **kwargs):
        """Function to compute W0, which is the integral
        of the kernel over the sphere."""
        
        self.__dict__.update(kwargs)
        
        #Do quadrature
        xi, weight = leg.leggauss(quad_ord)          
        self.W0 = 2.0*np.pi*sum(weight*self.kernel(xi))
        
        
    def test_leg(self, nmax=21, quad_ord=21):
        xi, weight = leg.leggauss(quad_ord)
        P = np.zeros((nmax, quad_ord))
        for i in range(nmax):
            poly = legendre(i)
            P[i,:] = np.sqrt(weight)*poly(xi)        
        return np.dot(P, P.T)
        
        
    def make_Wn(self, quad_ord=62, **kwargs):
        """Function that returns a vector of dimension nmax,
        where the nth element contains the integral of the
        product of the kernel with the nth Legendre polynomial."""
        
        self.__dict__.update(kwargs)
    
        #Make points and weights for Gauss-Legendre quadrature
        xi, weight = leg.leggauss(quad_ord)
        
        #Generate the n Legendre polynomials, evaluated at each point xi
        P = np.zeros((self.lmax+1, quad_ord))
        for i in range(self.lmax+1):
            poly = legendre(i)
            P[i,:] = weight*poly(xi)
        
        #Do quadrature for each polynomial order      
        self.Wn = 2.0*np.pi*np.dot(P, self.kernel(xi))
        
        
    def S(self, u, **kwargs):  
        self.__dict__.update(kwargs)        
        return 1.0/(1.0 + np.exp(-self.mu*(u - self.h)))
        
        
    def dS(self, u, **kwargs):
        self.__dict__.update(kwargs)
        s = self.S(u)
        return self.mu*s*(1.0 - s) 
        
        
    def d2S(self, u, **kwargs):        
        self.__dict__.update(kwargs)
        s = self.S(u)
        return (self.mu**2)*s*(1.0 - s)*(1.0 - 2.0*s)
        
        
    def scalar_eq(self, u, **kwargs):       
        self.__dict__.update(kwargs)  
        return -u + self.kappa*self.W0*self.S(u)
        
        
    def dispersion_curve(self, u, **kwargs):       
        self.__dict__.update(kwargs)             
        return -1.0 + self.kappa*self.Wn*self.dS(u)
        
                
    def stability_system(self, xi, p):        
        
        #Set parameters
        self.kappa = xi[1]
        self.h = p[0]
        self.mu = p[1]
        k = self.lambda_ord
        
        #Compute right hand side
        RHS = np.zeros(2)
        RHS[0] = self.scalar_eq(xi[0])
        RHS[1] = self.dispersion_curve(xi[0])[k]
        return RHS
        
        
    def stability_jac(self, xi, p):
    
        #Set parameters
        self.kappa = xi[1]
        self.h = p[0]
        self.mu = p[1]
        k = self.lambda_ord

        #Compute Jacobian
        Jac = np.zeros((2,2))
        Jac[0,0] = -1.0 + self.kappa*self.W0*self.dS(xi[0])
        Jac[0,1] = self.W0*self.S(xi[0])        
        Jac[1,0] = self.kappa*self.Wn[k]*self.d2S(xi[0])
        Jac[1,1] = self.Wn[k]*self.dS(xi[0])
        return Jac
        
        
    def make_u0(self, sigma=0.6, amp=5.0):
        
        #Make a Gaussian bump
        state = np.zeros((self.n, 2*self.n))
        for i in range(self.n):
            for j in range(2*self.n):
                state[i,j] = amp*np.exp(-((self.phi[i]-np.pi/2.0)**2 
                                        + (self.theta[j]-np.pi)**2) 
                                                    /(2.0*sigma**2))
        self.u0 = state 
        return self.u0
        
        
    def make_F(self, u, p):
    
        self.h = p[0]
        self.mu = p[1]
        umat = u.reshape(self.n, 2*self.n) 
        
        #Project firing rate into the harmonics
        s = self.S(umat)
        s_cilm = sh.SHExpandDH(s, sampling=2)
        
        #Multiply with wn
        result_cilm = np.zeros(np.shape(s_cilm))
        for i in range(2):
            for l in range(self.lmax + 1):
                for m in range(self.lmax + 1):
                    result_cilm[i,l,m] = self.Wn[l]*s_cilm[i,l,m]
                    
        #Reconstruct function back onto grid from the harmonics
        integral = sh.MakeGridDH(result_cilm, sampling=2)
        
        #Return operator
        temp = -umat + self.kappa*integral
        return temp.reshape(2*self.n*self.n) 
    
    
    def make_Jv(self, v, u, p):
    
        self.h = p[0]
        self.mu = p[1]
        umat = u.reshape(self.n, 2*self.n) 
        vmat = v.reshape(self.n, 2*self.n)
        
        #Project firing rate into the harmonics
        ds = self.dS(umat)*vmat
        s_cilm = sh.SHExpandDH(ds, sampling=2)
        
        #Multiply with wn
        result_cilm = np.zeros(np.shape(s_cilm))
        for i in range(2):
            for l in range(self.lmax + 1):
                for m in range(self.lmax + 1):
                    result_cilm[i,l,m] = self.Wn[l]*s_cilm[i,l,m]
                    
        #Reconstruct function back onto grid from the harmonics
        integral = sh.MakeGridDH(result_cilm, sampling=2)
        
        #Return operator
        temp = -vmat + self.kappa*integral
        return temp.reshape(2*self.n*self.n) 



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
     

def kernel(p, theta, phi):

    return im.make_routines.make_kernel(p, theta, phi)


def dist(phi1, theta1, phi2, theta2, radius=1):
    
    a = np.cos(phi1)*np.cos(phi2) + \
        np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2)
    return radius*np.arccos(a)

       
def make_u0(theta, phi, sigma=0.6, amp=5.0):
    
    #Make a Gaussian bump
    return amp*np.exp(-((phi - np.pi/2.0)**2 + (theta + np.pi/2.0)**2)  
                                                /(2.0*sigma**2))
    

def F(t, u, param, theta, phi, kern, w): 
    
    #Call the underlying Fortran routine make_F
    return im.make_routines.make_f(param,theta,phi,kern,w,u)
          
                  
def Jv(v, param, theta, phi, kern, w, u):
    
    #Call the underlying Fortran routine make_Jv
    return im.make_routines.make_jv(param,theta,phi,kern,w,u,v) 
    
    
def harmonicplot(theta, phi, u, un, fname, band_lim=20, threeD=False):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    plt.tight_layout()
    
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
                          
    
    #Render image as either a chart or a 3D image
    if threeD==False:
        ax.imshow(raster, cmap='viridis', norm=norm)
        ax.colorbar(shrink=0.5)
        ax.xlabel('$\\theta$',fontsize=26)
        ax.ylabel('$\phi$', fontsize=26, rotation=0, labelpad=15)
        ax.xticks([])
        ax.yticks([])
        ax.tight_layout()
        
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
        cbar = plt.colorbar(m, shrink=0.5) 
        
    plt.savefig(fname, format='jpg')
    ax.cla()
    plt.close()
        
        

    
    
        
        
        
        
        
        
        
