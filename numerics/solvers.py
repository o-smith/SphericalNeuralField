#! usr/bin/env python

import numpy as np
import scipy.sparse.linalg as alg
from numpy.linalg import norm


#Set up class to record convergence within GMRES
class KrylovCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = []
        self.resid = []
        self.counter = 0
    def __call__(self, rk=None):
        self.counter += 1
        self.niter.append(self.counter)
        self.resid.append(rk)
        # print self.counter, rk



def newton_gmres(jacfunc, func, u, p, toler=1e-5, restrt=40, gmres_tol=1e-5,
                nmax=8, gmres_max=1000, convobject=None, noisy=False):
    """This function computes the stationary solution to a generic  nonlinear
    problem. It uses matrix-free newton-gmres.
    """

    #Initialise some things
    n = len(u)
    counter = 0
    convergence = False

    #Make linear operator
    mv = lambda v: jacfunc(v, u, p)
    A = alg.LinearOperator((n,n), matvec=mv, dtype='float64')

    if convobject != None:
        errorreport = []

    while True:

        f = func(u, p) #Make right hand side

        #Call GMRES and make sure it converged
        if convobject != None:
            if counter == 0:
                x, info = alg.gmres(A, -f, tol=gmres_tol, restart=restrt,
                                    maxiter=gmres_max, callback=convobject)
            else:
                x, info = alg.gmres(A, -f, tol=gmres_tol, restart=restrt,
                                    maxiter=gmres_max)
        else:
            x, info = alg.gmres(A, -f, tol=gmres_tol, restart=restrt,
                                maxiter=gmres_max)
        if info > 0:
            return u, counter, info, convergence

        #Update solution
        u += x

        counter += 1
        if noisy:
            print "Newton iteration %i" %counter

        #Compute error and check convergence
        error = norm(x, ord=2)
        if convobject != None:
            errorreport.append(error)
        if (error < toler) :
            convergence = True
            break
        if counter > nmax :
            break

    if convobject != None:
        return u, counter, info, convergence, errorreport
    else:
        return u, counter, info, convergence 



def arnoldi(jacfunc, u, p, kk=1, toler=1e-5, what='LR',
             eigenvectors=False):
    """Function to compute eigenvalues of the Jacobian."""

    #Make linear operator
    n = len(u)
    mv = lambda v: jacfunc(v, u, p) 
    A = alg.LinearOperator((n,n), matvec=mv, dtype='float64')

    #Call scipy
    if eigenvectors == False :
        return alg.eigs(A, k=kk, which=what, tol=toler,
                  return_eigenvectors=False)[0]
    else:
        return alg.eigs(A, k=kk, which=what, tol=toler)



def number_of_pos(jacfunc, u, p, k=300, toler=1e-8):
    """Compute the number of positive eigenvalues"""

    #Make the linear operator
    n = len(u)
    mv = lambda v: jacfunc(v, u, p)
    A = alg.LinearOperator((n,n), matvec=mv, dtype="float64")

    #Compute the first k eigenvalues
    w = alg.eigs(A, k=k, which="LR", tol=toler, return_eigenvectors=False)

    #Scan for positive eigenvalues
    num_postitive = 0
    for i in range(k):
        if np.real(w[i]) > 0.0:
            num_postitive += 1
    return num_postitive



if __name__=="__main__":
	main() 


