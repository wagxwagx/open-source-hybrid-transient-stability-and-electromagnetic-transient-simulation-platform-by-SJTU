import numpy as np
nax = np.newaxis
from vectfit3 import vectfit3, tri2full, ss2pr
from time import time
import matplotlib.pyplot as plt


'''    These example scripts are part of the vector fitting package
       Last revised: 08.08.2008.
       Created by:    Bjorn Gustavsen.
       Translated by: Simon De Ridder                              '''


def ex1():
	''' Fitting an artificially created frequency response (single element)
	        -Creating a 3rd order frequency response f(s)
	        -Fitting f(s) using vectfit3.py
	            -Initial poles: 3 logarithmically spaced real poles
	            -1 iteration                                               '''
	# Frequency samples:
	Ns = 101
	s = 2 * np.pi * 1j * np.logspace(0,4,Ns)

	print('Creating frequency response f(s)...')
	f = np.empty((1,Ns), np.complex_)
	for k in range(Ns):
		f[0,k] = 2.0/(s[k]+5.0) + (30.0+1j*40.0)/(s[k]-(-100.0+1j*500.0))\
		                        + (30.0-1j*40.0)/(s[k]-(-100.0-1j*500.0)) + 0.5

	# Initial poles for Vector Fitting:
	N = 3 # order of approximation
	poles = -2.0 * np.pi * np.logspace(0,4,N) # Initial poles
	weight = np.ones((1,Ns)) # All frequency points are given equal weight

	opts = {}
	opts['relax'] = True      # Use vector fitting with relaxed non-triviality constraint
	opts['stable'] = True     # Enforce stable poles
	opts['asymp'] = 2         # Include both D, E in fitting    
	opts['skip_pole'] = False # Do NOT skip pole identification
	opts['skip_res'] = False  # Do NOT skip identification of residues (C,D,E) 
	opts['cmplx_ss'] = True   # Create complex state space model

	opts['spy1'] = False      # No plotting for first stage of vector fitting
	opts['spy2'] = True       # Create magnitude plot for fitting of f(s) 
	opts['logx'] = True       # Use logarithmic abscissa axis
	opts['logy'] = True       # Use logarithmic ordinate axis 
	opts['errplot'] = True    # Include deviation in magnitude plot
	opts['phaseplot'] = True  # Also produce plot of phase angle (in addition to magnitiude)
	opts['legend'] = True     # Do include legends in plots
	opts['block'] = True      # Block on plots

	print('vector fitting...')
	SER,poles,rmserr,fit = vectfit3(f,s,poles,weight,opts); 
	print('Done.')

	print('Resulting state space model:')
	print('A:', SER['A'])
	print('B:', SER['B'])
	print('C:', SER['C'])
	print('D:', SER['D'])
	print('E:', SER['E'])
	print('rms:', rmserr)


def ex2():
	''' -Creating an 18th order frequency response f(s) of 2 elements.
	    -Fitting f(s) using vectfit3
	    -Initial poles: 9 linearly spaced complex pairs (N=18)
	    -3 iterations                                                 '''
	D = 0.2
	E = 2.0e-5
	p = np.array([-4500.0, -41000.0,  (-100.0+1j*5.0e3),  (-100.0-1j*5.0e3),  (-120.0+1j*15.0e3),\
	              (-120.0-1j*15.0e3), (-3.0e3+1j*35.0e3), (-3.0e3-1j*35.0e3), (-200.0+1j*45.0e3),\
	              (-200.0-1j*45.0e3), (-1500.0+1j*45.0e3),(-1500.0-1j*45.0e3),(-5.0e2+1j*70.0e3),\
	              (-5.0e2-1j*70.0e3), (-1.0e3+1j*73.0e3), (-1.0e3-1j*73.0e3), (-2.0e3+1j*90.0e3),\
	              (-2.0e3-1j*90.0e3)])
	r = np.array([-3000.0, -83000.0, (-5.0+1j*7.0e3),   (-5.0-1j*7.0e3),   (-20.0+1j*18.0e3),\
	              (-20.0-1j*18.0e3), (6.0e3+1j*45.0e3), (6.0e3-1j*45.0e3), (40.0 +1j*60.0e3),\
	              (40.0-1j*60.0e3),  (90.0 +1j*10.0e3), (90.0-1j*10.0e3),  (5.0e4+1j*80.0e3),\
	              (5.0e4-1j*80.0e3), (1.0e3+1j*45.0e3), (1.0e3-1j*45.0e3), (-5.0e3+1j*92.0e3),\
	              (-5.0e3-1j*92.0e3)])
	p *= 2.0 * np.pi
	r *= 2.0 * np.pi

	w = 2.0 * np.pi * np.linspace(1,1e5,100)
	Ns = w.shape[0]
	s = 1j*w

	f = np.empty((2,Ns), dtype=np.complex_)
	f[0,:] = np.sum(r[nax,:10]/(s[:,nax]-p[nax,:10]), axis=1) + s*E + D
	f[1,:] = np.sum(r[nax,8:] /(s[:,nax]-p[nax,8:]),  axis=1) + s*3.0*E
	f[0,:] += 2.0 * D # should probably be f[1,:], but we'll stay true to original

	#Rational function approximation of f(s):
	N = 18 # Order of approximation

	# Complex starting poles:
	bet = np.linspace(w[0], w[-1], int(N/2))
	alf = -bet * 1.0e-2
	poles = np.concatenate(((alf-1j*bet)[:,nax],(alf+1j*bet)[:,nax]), axis=1).flatten()

	# Real starting poles:
	# poles = -np.linspace(w[0], w[-1], N)

	# Parameters for Vector Fitting:
	weight = np.ones((1,Ns))
	opts = {}
	opts['relax']     = True  # Use vector fitting with relaxed non-triviality constraint
	opts['stable']    = True  # Enforce stable poles
	opts['asymp']     = 2     # Include both D, E in fitting
	opts['skip_pole'] = False # Do NOT skip pole identification
	opts['skip_res']  = True  # DO skip identification of residues (C,D,E)
	opts['cmplx_ss']  = False # Create real-only state space model
	opts['spy1']      = False # No plotting for first stage of vector fitting
	opts['spy2']      = True  # Create magnitude plot for fitting of f(s)
	opts['logx']      = False # Use linear abscissa axis
	opts['logy']      = True  # Use logarithmic ordinate axis
	opts['errplot']   = True  # Include deviation in magnitude plot
	opts['phaseplot'] = False # Do NOT produce plot of phase angle
	opts['legend']    = True  # Include legends in plots
	opts['block']     = True  # Block on plots

	print('vector fitting...')
	Niter = 3
	rms = np.empty((Niter))
	for it in range(Niter):
		if it==Niter-1:
			opts['skip_res'] = False
		print('   Iter '+str(it))
		SER,poles,rmserr,fit = vectfit3(f, s, poles, weight, opts)
		rms[it] = rmserr
	print(rms)


def ex3():
	''' Fitting a measured admittance function from distribution transformer (single element)
	        -Reading frequency response f(s) from disk. (contains 1 element)
	        -Fitting f(s) using vectfit3
	        -Initial poles: 3 linearly spaced complex pairs (N=6)
	        -3 iterations                                                                    '''
	f = np.empty((1,150), dtype=np.complex_)
	with open('ZQD.txt','r') as fl:
		next(fl) # skip the first lines
		next(fl)
		for k in range(160):
			A1,A2 = [float(a) for a in fl.readline().strip().split()]
			f[0,k] =  (A1 * np.exp(1j*A2*1))

	w = 2.0 * np.pi * np.linspace(0,10e6,301)
	w = np.delete(w, 0)
	w = np.delete(w, np.s_[150:300])
	s = 1j * w
	Ns = s.shape[0]

	# Rational function approximation of f(s):
	N = 16 # Order of approximation

	# Complex starting poles:
	bet = np.linspace(w[0],w[-1],int(N/2))
	alf = -bet * 1.0e-2
	poles = np.concatenate(((alf-1j*bet)[:,nax],(alf+1j*bet)[:,nax]), axis=1).flatten()

	weight = np.ones((1,Ns)) # No weighting
	# weight = 1 / np.abs(f) # Weighting with inverse of magnitude function
	opts = {}
	opts['relax']     = True  # Use vector fitting with relaxed non-triviality constraint
	opts['stable']    = True  # Enforce stable poles
	opts['asymp']     = 2     # Include both D, E in fitting    
	opts['skip_pole'] = False # Do not skip pole identification
	opts['skip_res']  = False # Do not skip identification of residues (C,D,E) 
	opts['cmplx_ss']  = True  # Create real-only state space model
	opts['spy1']      = False # No plotting for first stage of vector fitting
	opts['spy2']      = False # Create magnitude plot for fitting of f(s) 
	opts['logx']      = False # Use linear abscissa axis
	opts['logy']      = True  # Use logarithmic ordinate axis 
	opts['errplot']   = True  # Include deviation in magnitude plot
	opts['phaseplot'] = True  # Include plot of phase angle
	opts['legend']    = False # do NOT include legends in plots
	opts['block']     = False # Block on plots

	print('vector fitting...')
	Niter = 5
	rms = np.empty((Niter))
	for it in range(Niter):
		if it==Niter-1:
			opts['spy2'] = True
			opts['legend'] = True # Include legend in final plot
		print('   Iter '+str(it))
		SER,poles,rmserr,fit = vectfit3(f,s,poles,weight,opts)
		rms[it] = rmserr
	print('RMS: ',rms)
	print('Done.')


def ex4a():
	''' Fitting 1st column of the admittance matrix of 6-terminal system  (power system
	    distribution network)
	       -Reading frequency admittance matrix Y(s) from disk.
	       -Extracting 1st column: f(s) (contains 6 elements)
	       -Fitting f(s) using vectfit3
	           -Initial poles: 25 linearly spaced complex pairs (N=50)
	           -5 iterations                                                           '''
	print('Reading data from file ...')
	with open('fdne.txt','r') as fid1:
		Nc = int(float(fid1.readline().strip()))
		Ns = int(float(fid1.readline().strip()))
		bigY = np.zeros((Nc,Nc,Ns), dtype=np.complex_)
		s = np.zeros((Ns), dtype=np.complex_)
		for k in range(Ns):
			s[k] = 1j * float(fid1.readline().strip())
			for row in range(Nc):
				for col in range(Nc):
					dum1 = float(fid1.readline().strip())
					dum2 = float(fid1.readline().strip())
					bigY[row,col,k] = dum1 + 1j*dum2

	# Extracting first column:
	f = bigY[:,0,:]
	start = time()
	print('-----------------S T A R T--------------------------')

	# Fitting parameters:
	N = 50 # Order of approximation
	# Complex starting poles:
	bet = np.linspace(s[0].imag, s[-1].imag, int(N/2))
	alf = -bet * 1.0e-2
	poles = np.concatenate(((alf-1j*bet)[:,nax],(alf+1j*bet)[:,nax]), axis=1).flatten()

	# weight = np.ones((Nc,Ns))
	# weight = 1 / np.abs(f)
	weight = 1 / np.sqrt(np.abs(f))

	# Fitting options
	opts = {}
	opts['relax']     = True # Use vector fitting with relaxed non-triviality constraint
	opts['stable']    = True # Enforce stable poles
	opts['asymp']     = 2
	opts['spy1']      = False
	opts['spy2']      = False
	opts['logx']      = False
	opts['logy']      = True
	opts['errplot']   = True
	opts['phaseplot'] = True
	opts['skip_pole'] = False
	opts['skip_res']  = False
	opts['cmplx_ss']  = False
	opts['legend']    = False

	Niter = 5
	rms = np.empty((Niter))
	print('*****Fitting column...')
	for it in range(Niter):
		print('   Iter '+str(it))
		if it==Niter-1:
			opts['spy2']   = True
			opts['legend'] = True
			opts['skip_pole'] = True
		SER,poles,rmserr,fit = vectfit3(f,s,poles,weight,opts)
		rms[it] = rmserr
	print('RMS: ', rms)
	print('-------------------E N D----------------------------')
	print('Elapsed: ', time()-start)

def ex4b():
	''' Fitting 1st column of the admittance matrix of 6-terminal system  (power system
	    distribution network)
	       -Reading frequency admittance matrix Y(s) from disk.
	       -Fitting g(s)=sum(f(s)) using vectfit3 --> new initial poles
	           -Initial poles: 25 linearly spaced complex pairs (N=50)
	           -5 iterations
	       -Fitting the six-columns one-by-one, stacking state-space model from each column into
	        global state-space model (3 iterations)
	       -Plotting of result                                                                  '''
	N = 50 # order of approximation
	Niter1 = 5 # Fitting column sum: n.o. iterations
	Niter2 = 3 # Fitting column: n.o. iterations

	print('Reading data from file ...')
	with open('fdne.txt','r') as fid1:
		Nc = int(float(fid1.readline().strip()))
		Ns = int(float(fid1.readline().strip()))
		bigY = np.zeros((Nc,Nc,Ns), dtype=np.complex_)
		s = np.zeros((Ns), dtype=np.complex_)
		for k in range(Ns):
			s[k] = 1j * float(fid1.readline().strip())
			for row in range(Nc):
				for col in range(Nc):
					dum1 = float(fid1.readline().strip())
					dum2 = float(fid1.readline().strip())
					bigY[row,col,k] = dum1 + 1j*dum2

	start = time()
	print('-----------------S T A R T--------------------------')
	bigf   = np.empty((Nc*Nc,Ns), dtype=np.complex_)
	bigfit = np.empty((Nc*Nc,Ns), dtype=np.complex_)

	# Complex starting poles:
	bet = np.linspace(s[0].imag, s[-1].imag,int(N/2))
	alf = -bet * 1.0e-2
	poles = np.concatenate(((alf-1j*bet)[:,nax],(alf+1j*bet)[:,nax]), axis=1).flatten()

	for col in range(Nc): # Column loop
		# Extracting elements in column:
		f = bigY[:,col,:]

		# Fitting options
		opts = {}
		opts['relax']     = True  # Use vector fitting with relaxed non-triviality constraint  
		opts['stable']    = True  # Enforce stable poles
		opts['asymp']     = 2     # Fitting includes D and E
		opts['spy1']      = False 
		opts['spy2']      = False 
		opts['logx']      = False 
		opts['logy']      = True 
		opts['errplot']   = True
		opts['phaseplot'] = False
		opts['skip_pole'] = False 
		opts['skip_res']  = True
		opts['cmplx_ss']  = False
		opts['legend']    = False

		if col==0:
			# First we fit the colum sum:
			# g = np.sum(f, axis=0)
			g = np.sum(f / np.linalg.norm(f, axis=1)[:,nax], axis=0)[nax,:]
			# g = np.sum(f / np.sqrt(np.linalg.norm(f, axis=1)[:,nax]), axis=0)
			weight_g = 1.0 / np.abs(g)

			print('****Improving initial poles by fitting column sum (1st column)...')
			for it in range(Niter1):
				print('   Iter '+str(it))
				if it==Niter1-1:
					opts['skip_res']=False
				SER,poles,rmserr,fit = vectfit3(g,s,poles,weight_g,opts)

		print('****Fitting column #'+str(col)+' ...')
		# weight = np.ones((1,Ns))
		# weight = 1.0 / np.abs(f)
		weight = 1.0 / np.sqrt(np.abs(f))
		if col==Nc-1:
			opts['legend']=True
		opts['skip_res']=True
		for it in range(Niter2):
			print('   Iter '+str(it))
			if it==Niter2-1:
				opts['skip_res']=False
				opts['spy2'] = True
			SER,poles,rmserr,fit = vectfit3(f,s,poles,weight,opts)

		# Stacking the column contribution into complete state space model:
		bigf[col*Nc:(col+1)*Nc,:]   = f
		bigfit[col*Nc:(col+1)*Nc,:] = fit

	# Finally, we assess the fitting quality (all matrix elements):
	freq = s.imag / (2.0*np.pi)
	plt.figure()
	h1 = plt.semilogy(freq, np.abs(bigf).T, 'b')
	h2 = plt.semilogy(freq, np.abs(bigfit).T,'r--')
	h3 = plt.semilogy(freq, np.abs(bigfit-bigf).T,'g-')
	plt.legend([h1[0],h2[0],h3[0]], ['Data','FRVF','Deviation'], loc='best')
	plt.xlabel('Frequency [Hz]')
	plt.title('All matrix elements')
	print('-------------------E N D----------------------------')
	print('Elapsed: ', time()-start)


def ex4c():
	''' Fitting 1st column of the admittance matrix of 6-terminal system  (power system
	    distribution network)
	       -Reading frequency admittance matrix Y(s) from disk.
	       -Stacking the (21) elements of the lower triangle of Y into a single column f(s)
	       -Fitting g(s)=sum(f(s)) --> new initial poles
	           -Initial poles: 25 linearly spaced complex pairs (N=50)
	           -5 iterations
	       -Fitting elements of vector f(s) using a common pole set (3 iterations)
	       -Converting resulting state-space model into a model for the full Y using tri2full
	       -Converting state space model (A,B,C,D,E) into a pole-residue model (a,R,D,E) using
	        ss2pr                                                                               '''
	N = 50 # order of approximation
	Niter1 = 5 # Fitting column sum: n.o. iterations
	Niter2 = 3 # Fitting column: n.o. iterations

	print('Reading data from file ...')
	with open('fdne.txt','r') as fid1:
		Nc = int(float(fid1.readline().strip()))
		Ns = int(float(fid1.readline().strip()))
		bigY = np.zeros((Nc,Nc,Ns), dtype=np.complex_)
		s = np.zeros((Ns), dtype=np.complex_)
		for k in range(Ns):
			s[k] = 1j * float(fid1.readline().strip())
			for row in range(Nc):
				for col in range(Nc):
					dum1 = float(fid1.readline().strip())
					dum2 = float(fid1.readline().strip())
					bigY[row,col,k] = dum1 + 1j*dum2

	start = time()
	print('-----------------S T A R T--------------------------')
	print('****Stacking matrix elements (lower triangle) into single column....')
	tell = 0
	f = np.empty((int(Nc*(Nc+1)/2),Ns), dtype=np.complex_)
	for col in range(Nc):
		f[tell:tell+Nc-col,:] = bigY[col:Nc,col,:] # stacking elements into a single vector
		tell += Nc - col
#		for row in range(col,Nc):
#			f[tell,:] = bigY[row,col,:] # stacking elements into a single vector
#			tell += 1

	# Complex starting poles:
	bet = np.linspace(s[0].imag, s[-1].imag, int(N/2))
	alf = -bet * 1.0e-2
	poles = np.concatenate(((alf-1j*bet)[:,nax],(alf+1j*bet)[:,nax]), axis=1).flatten()

	# weight = np.ones((1,Ns))
	# weight = 1 / np.abs(f)
	weight = 1 / np.sqrt(np.abs(f))

	# Fitting options
	opts = {}
	opts['relax']     = True  # Use vector fitting with relaxed non-triviality constraint
	opts['stable']    = True  # Enforce stable poles
	opts['asymp']     = 2     # Fitting includes D and E 
	opts['spy1']      = False
	opts['spy2']      = False
	opts['logx']      = False
	opts['logy']      = True
	opts['errplot']   = True
	opts['phaseplot'] = True
	opts['skip_pole'] = False
	opts['skip_res']  = True
	opts['cmplx_ss']  = True  # Will generate state space model with diagonal A
	opts['legend']    = True

	# Forming (weighted) column sum:
	# g = np.sum(f, axis=0)[nax,:] # unweighted sum
	g = np.sum(f / np.linalg.norm(f, axis=1)[:,nax], axis=0)[nax,:]
	# g = np.sum(f / np.sqrt(np.linalg.norm(f, axis=1)[:,nax]), axis=0)[nax,:]
	weight_g = 1 / np.abs(g)

	print('****Calculating improved initial poles by fitting column sum ...')
	for it in range(Niter1):
		print('   Iter '+str(it))
		if it==Niter1-1:
			opts['skip_res'] = False
			opts['spy2']     = True
		SER,poles,rmserr,fit = vectfit3(g,s,poles,weight_g,opts)

	print('****Fitting column ...')
	opts['skip_res'] = True
	opts['spy2']     = False
	for it in range(Niter2):
		print('   Iter '+str(it))
		if it==Niter2-1:
			opts['skip_res'] = False
			opts['spy2']     = True
		SER,poles,rmserr,fit = vectfit3(f,s,poles,weight,opts)

	print('****Transforming model of lower matrix triangle into state-space model of full matrix....')
	SER = tri2full(SER)
	print('****Generating pole-residue model....')
	R,a,D,E = ss2pr(SER, tri=True)
	print('-------------------E N D----------------------------')
	print('Elapsed: ', time()-start)


def ex5():
	''' The program approximates f(s) with rational functions. f(s) is a vector of 5 elements
	    represnting one column of the propagation matrix of a transmission line
	    (parallell AC and DC line). The elements of the prop. matrix have been backwinded using a
	    common time delay equal to the lossless time delay of the line.
	    -Reading frequency response f(s) from disk. (contains 5 elements)
	    -Fitting f(s) using vectfit3.m 
	    -Initial poles: 7 linearly spaced complex pairs (N=14)
	    -5 iterations                                                                           '''
	Ns = 60
	f = np.empty((5,Ns), dtype=np.complex_)
	s = np.empty((Ns), dtype=np.complex_)
	with open('w.txt','r') as fid1:
		with open('h.txt','r') as fid2:
			for k in range(Ns):
				s[k] = 1j * float(fid1.readline().strip())
				for n in range(5):
					a1,a2 = [float(a) for a in fid2.readline().strip().split()]
					f[n,k] = a1 + 1j*a2

	# Rational function approximation of f(s):
	N = 14 # Order of approximation

	# Complex starting poles:
	bet = np.linspace(s[0].imag, s[-1].imag, int(N/2))
	alf = -bet * 1.0e-2
	poles = np.concatenate(((alf-1j*bet)[:,nax],(alf+1j*bet)[:,nax]), axis=1).flatten()
	
	# Parameters for Vector Fitting:
	weight = np.ones((1,Ns))
	# weight = 1 / np.abs(f)

	opts = {}
	opts['relax']     = True  # Use vector fitting with relaxed non-triviality constraint
	opts['stable']    = True  # Enforce stable poles
	opts['asymp']     = 0     # Fitting with D=0, E=0
	opts['skip_pole'] = False # Do NOT skip pole identification
	opts['skip_res']  = False # Do NOT skip identification of residues (C,D,E)
	opts['cmplx_ss']  = True  # Create complex state space model
	opts['spy1']      = False # No plotting for first stage of vector fitting
	opts['spy2']      = False # Create magnitude plot for fitting of f(s)
	opts['logx']      = True  # Use logarithmic abscissa axis
	opts['logy']      = True  # Use logarithmic ordinate axis
	opts['errplot']   = True  # Include deviation in magnitude plot
	opts['phaseplot'] = True  # Also produce plot of phase angle (in addition to magnitiude)
	opts['legend']    = False # Do NOT include legends in plots

	Niter = 5
	rms = np.empty((Niter))
	for it in range(Niter):
		if it==Niter-1:
			opts['legend']=True
			opts['spy2']=True
		SER,poles,rmserr,fit = vectfit3(f,s,poles,weight,opts)
		rms[it] = rmserr
	print('RMS: ',rms)


if __name__=='__main__':
	ex3()