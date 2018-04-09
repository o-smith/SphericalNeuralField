#! usr/bin/env python 

import sys
# sys.path.append('/usr/lib64/python2.7/site-packages/pyshtools')
sys.path.append('/Users/oliversmith/iso_octo/SHTOOLS-3.1') 
# import _SHTOOLS as sh
import pyshtools as sh
import numpy as np
import numpy.polynomial.legendre as leg
import quadrature_rules as qr 
from scipy.special import legendre
from numerics.utilities import update_progress 
from numpy.linalg import norm


def greatcircledistance(phi1, theta1, phi2, theta2, radius):
    a = np.cos(phi1)*np.cos(phi2) + np.sin(phi1)*np.sin(phi2)*np.cos(theta1-theta2)
    return radius*np.arccos(a) 


class NeuralField:
    """
    Base class for neural fields

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

        #Default parameter settings 
        self.a1, self.b1, self.a2, self.b2 = 6.6, 1.0/28.0, 5.0, 1.0/20.0
        self.h, self.mu, self.kappa = 0.35, 8.0, 49.3155529412
        self.radius = 1.0 
        self.lambda_ord = 10
        self.lmax = 40 

        #Replace these defaults with user choices if necessary 
        self.__dict__.update(kwargs)


    def S(self, u, **kwargs):  
        """
        Function to compute the sigmoidal firing rate 
        """
        self.__dict__.update(kwargs)        
        return 1.0/(1.0 + np.exp(-self.mu*(u - self.h)))
        
        
    def dS(self, u, **kwargs):
        """
        Function to compute the derivative of the firing rate
        """
        self.__dict__.update(kwargs)
        s = self.S(u)
        return self.mu*s*(1.0 - s) 
        
        
    def d2S(self, u, **kwargs):    
        """
        Function to compute the second derivative of the firing rate 
        """    
        self.__dict__.update(kwargs)
        s = self.S(u)
        return (self.mu**2)*s*(1.0 - s)*(1.0 - 2.0*s)



class SphericalHarmonicNeuralField(NeuralField): 


    def __init__(self, **kwargs):
        NeuralField.__init__(self, **kwargs) 


    def makeGrid(self, **kwargs):
        self.__dict__.update(kwargs)
        
        #Construct the latitude-longitude grid 
        self.n = self.lmax*2 + 2
        latspacing = 180.0/float(self.n)
        lonspacing = 180.0/float(self.n)
        self.lat = np.arange(-90.0, 90.0, latspacing)
        self.lon = np.arange(0.0, 360.0, lonspacing)
        self.phi =  np.deg2rad(self.lat + 90.)
        self.theta = np.deg2rad(self.lon) 


    def finer_measure(self, u, l=40):
        """
        Function that takes a state u in column-major format and interpolates
        it onto a finer latitude-longitude grid, specifically the the same
        grid used in the interp_measure function. The function then 
        returns the value || u - <u> ||_2. 

        This function allows states computed in the harmonic method to 
        be measured on the same grid and with the same measure as those computed 
        using the quadrature schemes.
        """

        utemp = u.reshape((self.n, 2*self.n))
        cilm = sh.SHExpandDH(utemp, sampling=2) 
        ufine = sh.MakeGridDH(cilm, sampling=2, lmax=40) 
        mu = np.mean(ufine) 
        y = ufine - mu 
        return norm(y, ord=2) 


    def kernel(self, xi):
        """Function to compute kernel.
        The argument of this function is 
        xi = cos(theta)=<x,x'>, in [-1,1]."""
        
        d = np.arccos(xi)
        k = self.a1*np.exp(-(d**2)/self.b1) - self.a2*np.exp(-(d**2)/self.b2)
        return k
        
        
    def makeW0(self, quad_ord=60, **kwargs):
        """Function to compute W0, which is the integral
        of the kernel over the sphere."""
        
        self.__dict__.update(kwargs)
        
        #Do quadrature
        xi, weight = leg.leggauss(quad_ord)          
        self.W0 = 2.0*np.pi*sum(weight*self.kernel(xi))


    def makeWn(self, quad_ord=60, **kwargs):
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


    def scalarEq(self, u, **kwargs):       
        self.__dict__.update(kwargs)  
        return -u + self.kappa*self.W0*self.S(u)
        
        
    def dispersionCurve(self, u, **kwargs):       
        self.__dict__.update(kwargs)             
        return -1.0 + self.kappa*self.Wn*self.dS(u)
        
                
    def stabilitySystem(self, xi, p):        
        
        #Set parameters
        self.kappa = xi[1]
        self.h = p[0]
        self.mu = p[1]
        k = self.lambda_ord
        
        #Compute right hand side
        RHS = np.zeros(2)
        RHS[0] = self.scalarEq(xi[0])
        RHS[1] = self.dispersionCurve(xi[0])[k]
        return RHS
        
        
    def stabilityJac(self, xi, p):
    
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


    def make_u0(self, sigma=0.6, amp=5.0, **kwargs):
        self.__dict__.update(kwargs)
        
        #Make a Gaussian bump
        state = np.zeros((self.n, 2*self.n))
        for i in range(self.n):
            for j in range(2*self.n):
                state[i,j] = amp*np.exp(-((self.phi[i]-np.pi/2.0)**2 
                                        + (self.theta[j]-np.pi)**2) 
                                                    /(2.0*sigma**2))
        self.u0 = state 
        return self.u0


    def makeF(self, u, **kwargs):
        self.__dict__.update(kwargs) 
    
        umat = u.reshape(self.n, 2*self.n) 
        
        #Project firing rate into the harmonics
        s = self.S(umat)
        s_cilm = sh.SHExpandDH(s, sampling=2)
        
        #Multiply with Wn
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
    
    
    def makeJv(self, v, u, **kwargs):
        self.__dict__.update(kwargs) 
    
        umat = u.reshape(self.n, 2*self.n) 
        vmat = v.reshape(self.n, 2*self.n)
        
        #Project firing rate into the harmonics
        ds = self.dS(umat)*vmat
        s_cilm = sh.SHExpandDH(ds, sampling=2)
        
        #Multiply with Wn
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
       


class SphericalQuadratureNeuralField(NeuralField): 


    def __init__(self, **kwargs):
        NeuralField.__init__(self, **kwargs) 


    def makeGrid(self, rule, **kwargs):
        self.__dict__.update(kwargs) 
        self.rule = rule 
        if rule == "Lebedev" or rule == "lebedev":
            self.theta, self.phi, self.weights = qr.gen_grid() 
            self.n = len(self.theta) 
        elif rule == "Icosahedral" or rule == "icosahedral":
            self.theta, self.phi, self.weights = qr.generate_iso_grid('model/quadraturedata/qsph1-100-3432DP.txt')
        else:
            print "Quadrature rule %s not recognised" %rule 
            raise Exception
        self.n = len(self.theta) 


    def computeKernel(self, **kwargs):
        self.__dict__.update(kwargs)
        self.kernel = np.zeros((self.n, self.n)) 
        print "Computing kernel..."
        for i in range(self.n):
            update_progress(float(i)/float(self.n))
            for j in range(self.n):
                if i == j:
                    d = 0.0
                else:
                    d = greatcircledistance(self.phi[i], self.theta[i],
                        self.phi[j], self.theta[j], self.radius)
                self.kernel[i,j] = self.a1*np.exp(-d*d/self.b1) - self.a2*np.exp(-d*d/self.b2)
        print "\nDone" 

    def make_u0(self, sigma=0.6, amp=5.0, **kwargs):
        self.__dict__.update(kwargs) 
        return amp*np.exp(-((self.phi - np.pi/2.0)**2 + (self.theta - np.pi/2.0)**2)  
                                                /(2.0*sigma**2))


    def makeF(self, u, **kwargs):
        self.__dict__.update(kwargs) 
        Svec = self.weights*self.S(u)
        if self.rule == "lebedev" or self.rule == "Lebedev":
            return -u + 4.0*np.pi*self.kappa*np.dot(self.kernel, Svec)
        else:
            return -u + self.kappa*np.dot(self.kernel, Svec)


    def makeJv(self, v, u, **kwargs): 
        self.__dict__.update(kwargs) 
        dSvec = self.weights*v*self.dS(u) 
        if self.rule == "lebedev" or self.rule == "Lebedev":
            return -v + 4.0*np.pi*self.kappa*np.dot(self.kernel, dSvec)
        else:
            return -v + self.kappa*np.dot(self.kernel, dSvec)  



if __name__=="__main__":
    main()         
        

    
    
        
        
        
        
        
        
        
