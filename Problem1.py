# SQP Algorithm for HW5
# Benjamin Webb
# 11/21/2022

import numpy as np

def objfun(x):
	# objective function
	# x: 2x1 vector

	return x[0]**2 + (x[1] - 3.0)**2

def constraints(x):
	# vector of inequality constraints
	# x: 2x1 vector

	g = np.zeros((2, 1), dtype=np.single)
	g[0] = x[1]**2 - 2*x[0]
	g[1] = (x[1] - 1.0)**2 + 5.0*x[0] - 15.0

	return g

def gradConstraints(x):
	# matrix containing jacobian of the inequality constraints w.r.t. x
	# x: 2x1 vector

	dg = np.zeros((2, 2), dtype=np.single)
	dg[0, 0] = -2.0
	dg[0, 1] = 2*x[1]
	dg[1, 0] = 5.0
	dg[1, 1] = 2*x[1] - 2.0

	return dg

def gradLagrangian(x, mu):
	# Calculates gradient of the Lagrangian
	# x: 2x1 vector
	# g: 2x1 vector
	# k: interation number, int

	gradL = np.zeros((2, 1), dtype=np.single)
	gradL[0] = 2*x[0] - 2*mu[0] + 5*mu[1]
	gradL[1] = 2*x[1] - 6.0 + 2*mu[0]*x[1] + 2*mu[1]*x[1] - 2*mu[1]

	return gradL

def meritfun(x, mu, w, k):
	# merit function used in linesearch
	# x: 2x1 vector
	# mu: 2x1 vector containing lagrange multipliers
	# w: 2x2 matrix containing weights for merit function
	# k: iteration number

	if k == 0:
		w[:, 1] = np.abs(mu)
	else:
		w[:, 2] = np.maximum(np.abs(mu), 0.5*(w[:, 1] + np.abs(mu)))

	f = objfun(x)
	g = constraints(x)

	return f + np.sum(w[:, 2]*np.max(0.0, g))

def QP(x, mu, W):
	# Solves QP subproblem w/ active set
	# x: 2x1 vector
	# mu: 2x1 vector

	# Formulate intial set of active constraints
	A = gradConstraints(x)
	# if np.sign(mu[0]) <= 0.0:
	# 	A[0, 0] = 0.0
	# 	A[0, 1] = 0.0
	# if np.sign(mu[1]) <= 0.0:
	# 	A[1, 0] = 0.0
	# 	A[1, 1] = 0.0

	gbar = constraints(x)
	# if np.sign(mu[0]) <= 0.0:
	# 	gbar[0] = 0.0
	# if np.sign(mu[1]) <= 0.0:
	# 	gbar[1] = 0.0

	# Jacobian of objective function
	fx = np.array([[2*x[0]], [2*x[1] - 6.0]], dtype=np.single)
	# Jacobian of inequality constraints
	dg = gradConstraints(x)

	# Initial step size
	sk = np.zeros((2, 1), dtype=np.single)
	j = np.uint16(0)

	while j < 100:
		C = np.vstack((np.hstack((W, A.T)), np.hstack((A, np.zeros((2, 2), dtype=np.single)))))
		D = np.vstack((-fx, -gbar))

		# Least squares solution
		E = np.linalg.lstsq(C, D, rcond=None)

		# Update sk and mu
		sol = E[0]
		sk = sol[:2]
		mu = sol[2:4]
		dgdx1 = dg[0, 0:2]@sk + gbar[0]
		dgdx2 = dg[1, 0:2]@sk + gbar[1]

		# Update active constraints
		# A = gradConstraints(x)
		# gbar = constraints(x)
		if mu[0] <= 0.0:
			if mu[0] <= mu[1]:
				A[0, 0] = 0.0
				A[0, 1] = 0.0
				gbar[0] = 0.0
		if mu[1] <= 0.0:
			if mu[1] < mu[0]:
				A[1, 0] = 0.0
				A[1, 1] = 0.0
				gbar[1] = 0.0
		if mu[0] > 0.0 and mu[0] >= mu[1]:
			if dgdx1.max() > 0.0:
				A[0, 0] = -2.0
				A[0, 1] = 2*x[1]
				gbar[0] = x[1]**2 - 2*x[0]
		if mu[1] > 0.0 and mu[1] > mu[0]:
			if dgdx2.max() > 0.0:
				A[1, 0] = 5.0
				A[1, 1] = 2*x[1] - 2.0
				gbar[1] = (x[1] - 1.0)**2 + 5*x[0] - 15.0

		# Determine if QP subproblem is solved
		if mu[0] > 0 and mu[1] <= 0 or dgdx2.max() <= 0.0:
			if dgdx1.max() <= 0.0:
				break
		if mu[1] > 0 and mu[0] <= 0 or dgdx1.max() <= 0.0:
			if dgdx2.max() <= 0.0:
				break

		j += 1

	return sk, mu

def linesearch(sk, mu, ww, k):
	# Does linesearch for this SQP problem
	# Many variables needed are calculated inside function
	# sk: 2x1 vector of current step size
	# mu: 2x1 vector of current guess of Lagrange multipliers
	# ww: 2x2 matrix containing the merit function weights

	test = 1

if __name__ == "__main__":
	# main script

	# Iteration counter
	k = np.uint16(0)

	# Initial solution guess and step size
	sk = np.zeros((2, 1000), dtype=np.single)
	x = np.zeros((2, 1000), dtype=np.single)
	x[:, :1] = np.array([[1.0], [2.0]])

	# Determine if any of the inequality constraints are active
	g = np.zeros((2, 1000), np.single)
	g[:, :1] = constraints(x[:, k])
	mu = np.zeros((2, 1000), np.single)

	# Calculate gradient of Lagrangian at x0
	gradL = np.zeros((2, 1000), dtype=np.single)
	gradL[:, :1] = gradLagrangian(x[:, k], mu[:, k])

	# Initialize W
	W = np.zeros((2, 2, 1000), dtype=np.single)
	W[0, 0, 0] = 1.0
	W[1, 1, 0] = 1.0

	# Run QP
	[sk[:, k], mu[:, k]] = QP(x[:, k], mu[:, k], W[:, :, k])

	# Initialize array of weights to be used in linesearch
	ww = np.zeros((2, 1), 1000, dtype=np.single)
	ww[0, 0] = np.abs(mu[0, 0])
	ww[1, 0] = np.abs(mu[1, 0])

