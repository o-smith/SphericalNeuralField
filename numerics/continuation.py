#! usr/bin/env python 

import numpy as np 
import scipy.sparse.linalg as alg 
from numpy.linalg import norm 

def extended_system(jfunc, func, v, p, step, v1, output=1, w=None, epslon=1e-5):

	#Allocate space 
	n = len(v) - 1 
	F = np.zeros(n+1) 
	if output == 2:
		Jw = np.zeros(n+1) 

	#Construct the right hand side 
	F[:n] = func(v[:n]) 
	F[n] = np.dot(tang, (v-v1)) - step 
	if output == 1:
		return F 

	#Construct extended Jacobian-vector product 
	if output == 2:
        Jw[:n] = Jfunc(w[:n], v[:n])
        pert = v[n] + epslon
        Jw[:n] += w[n]*(func(v[:n], p=pert) - F[:n])/epslon
        Jw[n] = np.dot(tang, w)
        return Jw


def stability(jacfunc, u, tolerance=1e-5):
    """Function to assess the stability of a solution by evaluating the
    largest eigenvalues of the Jacobian evaluated at that solution. This
    function then returns a value of 0 if the largest eigenvalue is negative
    (denoting stability) or 1 if not (denoting instability).
    """

    y = arnoldi(jacfunc, u, toler=tolerance)
    if y > 0.0 :
        return 1
    else:
        return 0


def secant_continuation(Jfunc, func, measure,
                       u, p, txtfilename=None,
                       init_direction=-1.0, soloutfreq=20, solout=False,
                       whichpar=1, soltol=1e-4, eigtol=1e-5, newtonmax=8,
                       optimalcount=4, epsilon=1e-5,
                       llimit=None, ulimit=None, count_max=None,
                       step_size=0.1, step_factorup=2.0,
                       step_factordown=2.0, plotting=False,
                       step_min=1e-5, step_max=0.25, krylov_max=50,
                       stability_analysis=True):


    #Check to make sure there is some stopping criteria
    if (llimit == None) and (ulimit == None) and (count_max == None):
        print 'Warning: no stopping criteria.'
        print 'Program will exit when/if a solution cannot be found.'
        print 'Or press ctrl-c to discontinue at any point.'

    #Open file for recording
    s = 0
    if txtfilename != None:
        wr = open(txtfilename,"w")
    with open(filename,'w') as output:

        #This try statement allows user to cleanly discontinue
        #at any point using keyboard interruption
        try:

            #Find an initial branch point
            n = len(u)
            u0 = newton_gmres(Jfunc, func, u, p, toler=soltol,
                              nmax=50,
                              gmres_max=200)[0]

            #Make extended vector
            xi0 = np.append(u0, p[whichpar])

            #Compute norm and stability
            print 'Computing stability...'
            if stability_analysis:
                s = stability(Jfunc, xi0[:n], p, tolerance=eigtol)
            y = measure(xi0[:n])
            print 'Continuation parameter =', p[whichpar]
            print 'Norm =', y
            print 'Stability =', s

            #Save the full solution
            soloutfilename = 'kernel2bifs/octo/isola/state_%f' %p[whichpar]
            soloutfilename += '_%f' %y
            soloutfilename += '.txt'
            if solout:
                np.savetxt(soloutfilename, xi0[:n])

            #Record this data point
            if txtfilename != None:
                outyvector = (p[whichpar], y, float(s))
                wr.write(" ".join(map(str,outyvector))+"\n")

            #Do a step of natural parameter continuation
            p[whichpar] += 0.001*init_direction
            u1 = newton_gmres(Jfunc, func, xi0[:n], p, toler=soltol,
                              nmax=newtonmax, gmres_max=krylov_max)[0]

            #Make another extended vector
            xi1 = np.append(u1, p[whichpar])

            #Compute norm and stability
            print 'Computing stability...'
            if stability_analysis:
                s = stability(Jfunc, xi1[:n], p, tolerance=eigtol)
            y = measure(xi1[:n])
            print 'Continuation parameter =', p[whichpar]
            print 'Norm =', y
            print 'Stability =', s

            #Record this data point
            if txtfilename != None:
                outyvector = (p[whichpar], y, float(s))
                wr.write(" ".join(map(str,outyvector))+"\n")

            #So we now have two previous points, xi0 and xi1
            step = step_size
            tang = (xi1-xi0)/norm(xi1-xi0, ord=2)

            #Make "function handles" to construct the extended problem
            Gfunc  = lambda u, p: ExtendedSystem(Jfunc, func, u, tang, p,
                                    step, xi1, output=1, whichpar=whichpar)
            dGfunc = lambda w, u, p: ExtendedSystem(Jfunc, func, u,
                                    tang, p, step, xi1, output=2, w=w,
                                    whichpar=whichpar,
                                    epslon=1e-5)

            #Start prediction-correction
            counter = 0
            while True:

                #Make tangent vector
                tang = (xi1-xi0)/norm((xi1-xi0), ord=2)

                #Make prediction
                xi = xi1 + step*tang
                print 'step =', step

                #Do correction step
                try:
                    tmp, ncounter, ninfo, conv = newton_gmres(
                            dGfunc, Gfunc, xi, p, toler=soltol,
                            nmax=newtonmax, gmres_max=krylov_max)
                except OverflowError:
                    conv = False

                #Further checks to convergencce
                if abs(norm(tmp) - norm(xi1)) > 2.0:
                    conv = False
                if (abs(tmp[n] - xi1[n])) > 0.5:
                    conv = False

                if conv: #If Newton-GMRES converged, update solution
                    xi = tmp
                    p[whichpar] = xi[n]

                    #Compute norm and stability
                    print 'Computing stability...'
                    if stability_analysis:
                        s = stability(Jfunc, xi[:n], p, tolerance=eigtol)
                    y = measure(xi[:n])
                    print 'Continuation parameter =', p[whichpar]
                    print 'Norm =', y
                    print 'Stability =', s
                    print 'Counter =', counter

                    #Record this branch point
                    if txtfilename != None:
                        outyvector = (p[whichpar], y, float(s))
                        wr.write(" ".join(map(str,outyvector))+"\n")

                    #Periodically record full solution
                    if (counter % soloutfreq == 0):
                        soloutfilename = ''
                        soloutfilename = 'kernel2bifs/octo/isola/state_%f'  %p[whichpar]
                        soloutfilename += '_%f' %y
                        soloutfilename += '.txt'
                        if solout:
                            np.savetxt(soloutfilename, xi[:n])

                    #Update solution history
                    xi0 = xi1
                    xi1 = xi
                    counter += 1

                    #Adjust step size
                    if ncounter > optimalcount:
                        if step > step_min:
                            step = step/step_factordown
                        else:
                            print 'Warning: min step size reached.'
                            print 'Exiting...'
                            if txtfilename != None:
                                wr.close()
                            return xi[:n]
                    if ncounter < optimalcount:
                        if step < step_max:
                            step = step*step_factorup

                    #Check stopping criteria
                    if ulimit != None:
                        if p[whichpar] > ulimit :
                            print 'Upper limit of continuation reached.'
                            print 'Exiting...'
                            if txtfilename != None:
                                wr.close()
                            return xi[:n]
                    if llimit != None:
                        if p[whichpar] < llimit :
                            print 'Lower limit of continuation reached.'
                            print 'Exiting...'
                            if txtfilename != None:
                                wr.close()
                            return xi[:n]
                    if count_max != None:
                        if counter > count_max:
                            print 'Maximum count reached.'
                            print 'Exiting...'
                            if txtfilename != None:
                                wr.close()
                            return xi[:n]

                else: #If Newton-GMRES did not converge
                    if step > step_min:
                        step = step/step_factordown
                    else:
                        print 'Min step size reached without convergence.'
                        print 'Exiting...'
                        if txtfilename != None:
                            wr.close()
                        return xi[:n]

        #Handle errors and user interruptions
        except Exception:
            print ' Exiting...'
            if txtfilename != None:
                wr.close()
            sys.exit(0)
        except KeyboardInterrupt:
            print ' Exiting...'
            if txtfilename != None:
                wr.close()
            sys.exit(0)



if __name__=="__main__":
	main() 























