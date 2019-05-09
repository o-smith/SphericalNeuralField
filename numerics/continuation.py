#! usr/bin/env python 

import sys
import numpy as np 
import scipy.sparse.linalg as alg 
from numpy.linalg import norm 
from numerics.solvers import *
from numerics.bif import * 


def bisect_targeter(h, target):
    if h > target:
        return 0
    else:
        return 1


def extended_system(output, jfunc, func, v, p, step, tang, v1, w=None, **kwargs):

	epslon = kwargs.get("epslon", 1e-5)
	whichpar = kwargs.get("whichpar", 0)

	#Allocate space 
	n = len(v) - 1 
	p[whichpar] = v[n]
	F = np.zeros(n+1) 
	if output == 2:
		Jw = np.zeros(n+1) 

	#Construct the right hand side 
	F[:n] = func(v[:n], p) 
	F[n] = np.dot(tang, (v-v1)) - step 
	if output == 1:
		return F 

	#Construct extended Jacobian-vector product 
	if output == 2:
		Jw[:n] = jfunc(w[:n], v[:n], p)
		p[whichpar] = v[n] + epslon
		Jw[:n] += w[n]*(func(v[:n], p) - F[:n])/epslon
		Jw[n] = np.dot(tang, w)
		return Jw


def right_hand_border(func, v, p, **kwargs):

	epslon = kwargs.get("epslon", 1e-5)
	whichpar = kwargs.get("whichpar", 0)

	#Compute difference
	F = func(v, p)
	p[whichpar] += epslon
	F2 = func(v, p)
	delt = (F2 - F)/epslon
	return delt


def bisect(jfunc, func, measure, u, p, **kwargs): 

	init_direction = kwargs.get("init_direction", -1)
	whichpar = kwargs.get("whichpar", 1)
	stability_bisect = kwargs.get("stability_bisect", False)
	target = kwargs.get("target", 0.35) 

	#Vectors to store things
	n = len(u) 
	uvector = np.zeros(n)
	xi = np.zeros(n+1)
	xi0 = np.zeros(n+1)
	xi1 = np.zeros(n+1)

	#Do simple parameter continuation of the resting state
	targ_old = 0
	deltah = 0.1
	counter = 0
	print measure(u)

	#Solve equation and make extended vector
	u0, ncounter, ninfo, conv = newton_gmres(
				jfunc, func, u, p, toler=1e-10,
				nmax=30, gmres_max=100)
	xi0[:n] = u0
	xi0[n] = p[whichpar]

	#Compute stability
	print 'Arnoldi...'
	x = stability(jfunc, xi0[:n], p, tolerance=1e-1)

	#Output data
	normout = measure(xi0[:n])
	print p[whichpar], normout, x

	#Do one step of natural parameter continuation
	p[whichpar] -= deltah
	u1, ncounter, ninfo, conv = newton_gmres(
				jfunc, func, xi0[:n], p, toler=1e-10,
				nmax=30, gmres_max=100)
	xi1[:n] = u1
	xi1[n] = p[whichpar]

	#Compute stability
	print 'Arnoldi...'
	x = stability(jfunc, xi1[:n], p, tolerance=1e-1)

	#Output data
	normout = measure(xi1[:n])
	print p[whichpar], normout, x

	#Begin bisection
	x_old = x 
	targ_old = 0 
	print 'Beginning bisection...'
	while abs(deltah) > 1e-5:

		#Make tangent vector
		tang = (xi1 - xi0)/norm((xi1 - xi0), ord=2)

		#Solve scalar equation
		p[whichpar] -= deltah
		utemp, ncounter, ninfo, conv = newton_gmres(
				jfunc, func, xi1[:n], p, toler=1e-6,
				nmax=30, gmres_max=400)
		if conv==False:
			print '\nOpps\n'
			break

		#Make extended vector
		xi[:n] = utemp
		xi[n] = p[whichpar]

		#Remember previous two solutions
		np.copyto(xi0, xi1)
		np.copyto(xi1, xi)
		counter += 1

		#Compute stability
		print 'Arnoldi...'
		if stability_bisect:
			x = stability(jfunc, xi[:n], p, tolerance=1e-1)
		print p[whichpar]

		#Bisect if target has been passed
		if stability_bisect:
			if x != x_old:
				deltah = -deltah/2.0
				print 'Bisecting!'
				print 'deltah = %f' %deltah
			x_old = x 
		else:
			targ = bisect_targeter(p[whichpar], target) 
			if targ != targ_old:
				deltah = -deltah/2.0
				print 'Bisecting!'
				print 'deltah = %f' %deltah
			targ_old = targ

	return xi1, xi0, p 



def stability(jacfunc, u, p, tolerance=1e-5):
    """Function to assess the stability of a solution by evaluating the
    largest eigenvalues of the Jacobian evaluated at that solution. This
    function then returns a value of 0 if the largest eigenvalue is negative
    (denoting stability) or 1 if not (denoting instability).
    """

    y = arnoldi(jacfunc, u, p, toler=tolerance)
    if y > 0.0 :
        return 1
    else:
        return 0


def branch_switch_secant_continuation(Jfunc, func, measure, xi1, xi0, eigval, p, **kwargs):

	txtfilename = kwargs.get("txtfilename", None)
	init_direction = kwargs.get("init_direction", -1.0)
	whichpar = kwargs.get("whichpar", 0)
	soltol = kwargs.get("soltol", 1e-3)
	eigtol = kwargs.get("eigtol", 1e-1)
	netwtonmax = kwargs.get("newtonmax", 16)
	optimalcount = kwargs.get("optimalcount", 4)
	epsilon = kwargs.get("epsilon", 1e-5)
	llimit = kwargs.get("llimit", None)
	ulimit = kwargs.get("ulimit", None)
	count_max = kwargs.get("count_max", None)
	step_size = kwargs.get("step_size", 0.1)
	step_factorup = kwargs.get("step_factorup", 1.5)
	step_factordown = kwargs.get("step_factordown", 2.0)
	step_min = kwargs.get("step_min", 1e-5)
	step_max = kwargs.get("step_max", 0.2)
	krylov_max = kwargs.get("krylov_max", 40)
	stability_analysis = kwargs.get("stability_analysis", True) 


 	#Eigen direction in which to perturb the continuation                       
	evec, evecT = eigval, eigval 

	#Make the first tangent vector based on two previous points
	tang = (xi1 - xi0)/norm((xi1 - xi0), ord=2)

	#First do step 2 from pde2path paper
	n = len(xi1) - 1 
	alpha_0 = tang[n]
	alpha_1 = np.dot(evecT, tang[:n])
	phi_0 = (tang[:n] - alpha_1*evec)/alpha_0

	#Now do step 3 from pde2path (calculate finite differences)
	pert = xi1[:n] + epsilon*evec
	jvtmp1 = Jfunc(evec, pert, p)
	jvtmp2 = Jfunc(evec, xi1[:n], p)
	jvtmp1 -= jvtmp2
	a1 = np.dot(evecT, jvtmp1)/epsilon 

	jvtmp1 = jfunc(phi_0, pert, p)
	jvtmp2 = jfunc(phi_0, xi1[:n], p)
	dhtmp1 = right_hand_border(func, pert, p, epslon=epsilon)
	dhtmp2 = right_hand_border(func, xi1[:n], p, epslon=epsilon)
	jvtmp1 = jvtmp1 - jvtmp2 + dhtmp1 - dhtmp2
	b1 = np.dot(evecT, jvtmp1)/diff

	#Do final step from p2p
	alpha_1bar = a1*alpha_1/alpha_0
	alpha_1bar = -alpha_1bar - 2.0*b1

	#Finally, set the new tangent
	tau = np.zeros(n+1)
	tau[:n] = alpha_1bar*evec + a1*phi_0
	tau[n] = a1
	tang = -tau/norm(tau, ord=2)

	#Make "function handles" to construct the extended problem
	Gfunc  = lambda u, p: extended_system(1, Jfunc, func, u, p,
							step, tang, xi1, whichpar=whichpar)
	dGfunc = lambda w, u, p: extended_system(2, Jfunc, func, u,
							p, step, tang, xi1, w=w,
							whichpar=whichpar,
							epslon=1e-5)

	# Do secant continuation along new branch
	ulimit, count_max = None, None
	counting = 0
	x = 1
	try:
		counter = 0
		while True:

			#Set new vector
			if counting > 0:
				tang = (xi1-xi0)/norm((xi1-xi0), ord=2)
			else:
				print 'New tangent time!'

			#Make prediction
			xi = xi1 + step*tang
			print 'Step =', step

			#Do correction step
			try:
				tmp, ncounter, ninfo, conv = newton_gmres(
						dGmake_F, Gmake_F, xi, p, toler=1e-6,
						nmax=6, gmres_max=40)
			except OverflowError:
				print '\nOverfolw error encountered...\n'
				conv = False

			if conv: #If Newton-GMRES converged, update solution
				xi = tmp
				p[1] = xi[n]

				#Compute norm and stability
				if counting > 20:
					print 'Computing stability...'
					x = stability(jfunc, xi[:n], p, tolerance=1e-1)
				u2 = measure(xi[:n])
				print 'Continuation parameter =', p[whichpar]
				print 'Norm =', u2
				print 'Stability =', x
				print 'Counter =', counting

				#Record and plot this point
				outyvector = (p[whichpar], u2, float(x))
				wr.write(" ".join(map(str,outyvector))+"\n")

				#Update solution history
				np.copyto(xi0, xi1)
				np.copyto(xi1, xi)
				counting += 1

				#Adjust step size
				if ncounter > optimal:
					if step > step_min:
						step = step/step_factordown
					else:
						print 'Warning: min step size reached.'
						print 'Exiting...'
						wr.close()
						break
				if ncounter < optimal:
					if step < step_max:
						step = step*step_factorup

				#Check stopping criteria
				if ulimit != None:
					if p[whichpar] > ulimit :
						print 'Upper limit of continuation reached.'
						print 'Exiting...'
						wr.close()
						break
				if llimit != None:
					if p[whichpar] < llimit :
						print 'Lower limit of continuation reached.'
						print 'Exiting...'
						wr.close()
						break
				if count_max != None:
					if counter > count_max:
						print 'Maximum count reached.'
						print 'Exiting...'
						wr.close()
						break

			else: #If Newton-GMRES did not converge
				if step > step_min:
					step = step/step_factordown 
				else:
					print 'Min step size reached without convergence.'
					print 'Exiting...'
					wr.close()
					break

	#Handle errors and user interruptions
	except Exception as e:
		print e 
		print 'Exiting...'
		sys.exit(0)
	except KeyboardInterrupt:
		print ' Exiting...'
		sys.exit(0)


 
def secant_continuation(Jfunc, func, measure, u, p, **kwargs): 

	txtfilename = kwargs.get("txtfilename", None)
	init_direction = kwargs.get("init_direction", -1.0)
	whichpar = kwargs.get("whichpar", 0)
	soltol = kwargs.get("soltol", 1e-3)
	eigtol = kwargs.get("eigtol", 1e-1)
	netwtonmax = kwargs.get("newtonmax", 16)
	optimalcount = kwargs.get("optimalcount", 4)
	epsilon = kwargs.get("epsilon", 1e-5)
	llimit = kwargs.get("llimit", None)
	ulimit = kwargs.get("ulimit", None)
	count_max = kwargs.get("count_max", None)
	step_size = kwargs.get("step_size", 0.1)
	step_factorup = kwargs.get("step_factorup", 1.5)
	step_factordown = kwargs.get("step_factordown", 2.0)
	step_min = kwargs.get("step_min", 1e-5)
	step_max = kwargs.get("step_max", 0.2)
	krylov_max = kwargs.get("krylov_max", 40)
	stability_analysis = kwargs.get("stability_analysis", True) 


	#Check to make sure there is some stopping criteria
	if (llimit == None) and (ulimit == None) and (count_max == None):
		print 'Warning: no stopping criteria.'
		print 'Program will exit when/if a solution cannot be found.'
		print 'Or press ctrl-c to discontinue at any point.'

	#Open file for recording
	s = 0
	if txtfilename != None:
		wr = open(txtfilename,"w")

	#This try statement allows user to cleanly discontinue
	#at any point using keyboard interruption
	try:

		#Find an initial branch point
		n = len(u)
		u0 = newton_gmres(Jfunc, func, u, p, toler=1e-14,
						nmax=50,
						gmres_max=200)[0]

		#Make extended vector
		xi0 = np.append(u0, p[whichpar])

		#Compute norm and stability
		print 'Computing stability...'
		if stability_analysis:
			s = stability(Jfunc, xi0[:n], p, tolerance=eigtol)
		y = measure(xi0[:n])
		print 'Continuation parameter = %f' %p[whichpar]  
		print 'Norm = %f' %y
		print 'Stability = %i' %s

		#Record this data point
		if txtfilename != None:
			outyvector = (p[whichpar], y, float(s))
			wr.write(" ".join(map(str,outyvector))+"\n")

		#Do a step of natural parameter continuation
		p[whichpar] += 0.01*init_direction
		u1 = newton_gmres(Jfunc, func, xi0[:n], p, toler=1e-14,
						nmax=newtonmax, gmres_max=krylov_max)[0]

		#Make another extended vector
		xi1 = np.append(u1, p[whichpar])

		#Compute norm and stability
		print 'Computing stability...'
		if stability_analysis:
			s = stability(Jfunc, xi1[:n], p, tolerance=eigtol)
		y = measure(xi1[:n])
		print 'Continuation parameter = %f' %p[whichpar]
		print 'Norm = %f' %y
		print 'Stability = %i' %s

		#Record this data point
		if txtfilename != None:
			outyvector = (p[whichpar], y, float(s))
			wr.write(" ".join(map(str,outyvector))+"\n")

		#So we now have two previous points, xi0 and xi1
		step = 0.01
		tang = (xi1-xi0)/norm(xi1-xi0, ord=2)

        #Make "function handles" to construct the extended problem
		Gfunc  = lambda u, p: extended_system(1, Jfunc, func, u, p,
						step, tang, xi1, whichpar=whichpar)
		dGfunc = lambda w, u, p: extended_system(2, Jfunc, func, u,
						p, step, tang, xi1, w=w,
						whichpar=whichpar,
						epslon=1e-5)

		#Start prediction-correction
		counter = 0
		p_out, u_out, s_out = [], [], [] 
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

			if conv: #If Newton-GMRES converged, update solution
				xi = tmp
				p[whichpar] = xi[n]

				#Compute norm and stability
				print 'Computing stability...'
				if stability_analysis:
					s = stability(Jfunc, xi[:n], p, tolerance=eigtol)
				y = measure(xi[:n])
				print 'Continuation parameter = %f' %p[whichpar]
				print 'Norm = %f' %y
				print 'Stability = %i' %s
				print 'Counter = %i' %counter
				p_out.append(p[whichpar])
				u_out.append(y)
				s_out.append(s) 
				if counter%10 == 0:
					plot_bif(p_out, u_out, s_out) 

				#Record this branch point
				if txtfilename != None:
					outyvector = (p[whichpar], y, float(s))
					wr.write(" ".join(map(str,outyvector))+"\n")

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
						step *= step_factorup

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
					step /= step_factordown
				else:
					print 'Min step size reached without convergence.'
					print 'Exiting...'
					if txtfilename != None:
						wr.close()
					return xi[:n]

	#Handle errors and user interruptions
	except Exception as e:
		print e 
		print 'Exiting...'
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























