import numpy as np
import numpy.linalg as la
from mnorm import * 


def gen_orth_factors(s,R,eps = 1e-03):
	factors = []
	for i in range(3):
		A = np.random.uniform(low=-1,high=1, size =(s,int(R/2)))
		Q,_ = la.qr(A)
		B = (np.eye(s) - Q@Q.T)@np.random.uniform(low=-1,high=1, size =(s,int(R/2))) + eps*np.random.randn(s,int(R/2))
		A = A/la.norm(A,axis=0)
		B = B/la.norm(B,axis=0)
		#print(la.norm(A,axis=0))
		#print(la.norm(B,axis=0))
		factors.append(np.hstack((A,B)))
	return factors

def gen_guess(factors,noise = 1e-06):
	guess =[]
	for i in range(3):
		A = factors[i][:,:int(R/2)] + noise*np.random.randn(s,int(R/2))
		guess.append(A)
	return guess

def check_factors(factors,U,V,W):
	approx_factors = [U,V,W]
	d = 0
	for i in range(len(factors)):
		d += scalinvar_distance(factors[i][:,:int(R/2)],approx_factors[i])
	return d

s = 10
R = 10
num_gen = 100
num_init = 5
conv_tol = 1e-9
total = []
eps_perp = [1.e-05, 2.e-05, 3.e-05, 4.e-05, 5.e-05, 6.e-05, 7.e-05, 8.e-05,
       9.e-05, 1.e-04]

#eps_perp = [3.16227766e-05, 3.20754560e-05, 3.25346155e-05, 3.30003479e-05,
#       3.34727472e-05, 3.39519089e-05, 3.44379298e-05, 3.49309081e-05,
#       3.54309434e-05, 3.59381366e-05]
means = []
for eps in eps_perp:
	total = []
	for i in range(num_gen):
		factors =  gen_orth_factors(s,R,eps)
		T = np.einsum('ir,jr,kr->ijk',factors[0],factors[1],factors[2],optimize=True)
		converged = 0
		for j in range(num_init):
			[U,V,W] = gen_guess(factors,noise = 0.1)
			for k in range(5000):
				[U,V,W],conv = mnorm_iter(T,U,V,W)
				if conv:
					#print('iterations taken',k)
					break
			d = check_factors(factors,U,V,W)
			if d <= conv_tol :
				converged += 1
		#print('for num gen',i,'number of iterations converged is',converged)
		total.append(converged)
	print('for eps',eps,'iterations conv are',total)
	print('mean is',np.mean(total))
	means.append(np.mean(total))

print(means)




