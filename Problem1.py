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
	dg[0, 0] = - 2.0
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
	# w: 2xn vector containing weights for merit function
	# k: iteration number

	if k == 0:
		w[:, k] = np.zeros((2, 1), dtype=np.single)
	else:
		w[:, k] = np.maximum(mu[:, k], 0.5*(w[:, k-1] + np.abs(mu[:, k])))

	f = objfun(x)
	g = constraints(x)

	return f + np.sum(w[:, k]*np.max(0.0, g))

def QP(x, mu, W):
	# Solves QP subproblem w/ active set
	# x: 2x1 vector
	# mu: 2x1 vector

	# Formulate intial set of active constraints
	A = gradConstraints(x)
	if np.sign(mu[0]) <= 0.0:
		A[0, 0] = 0.0
		A[0, 1] = 0.0
	if np.sign(mu[1]) <= 0.0:
		A[1, 0] = 0.0
		A[1, 1] = 0.0

	gbar = constraints(x)
	if np.sign(mu[0]) <= 0.0:
		gbar[0] = 0.0
	if np.sign(mu[1]) <= 0.0:
		gbar[1] = 0.0

	# Jacobian of objective function
	fx = np.array([[2*x[0]], [2*x[1] - 6.0]], dtype=np.single)

	# Initial step size
	sk = np.zeros((2, 1), dtype=np.single)

	while mu.max() <= 0.0 and np.max(A@sk + gbar) > 0.0:
		C = np.vstack((np.hstack((W, A.T)), np.hstack((A, np.zeros((2, 2), dtype=np.single)))))
		D = np.vstack((-fx, -gbar))

		# Least squares solution
		E = np.linalg.lstsq(C, D, rcond=None)

		# Update sk and mu
		sol = E[0]
		sk = sol[:2]
		mu = sol[2:4]
		gbar = constraints(x+sk)

		# Update active constraints
		if mu[0] <= 0.0:
			if mu[0] < mu[1]:
				A[0, 0] = 0.0
				A[0, 1] = 0.0
				gbar[0] = 0.0
		if mu[1] <= 0.0:
			if mu[1] < mu[0]:
				A[1, 0] = 0.0
				A[1, 1] = 0.0
				gbar[1] = 0.0
		if np.min(mu) <= 0.0:
			idx = np.argmin(mu)
			if idx == 0:
				A[0, 0] = 0.0
				A[0, 1] = 0.0
				gbar[0] = 0.0
			elif idx == 1:
				A[1, 0] = 0.0
				A[1, 1] = 0.0
				gbar[1] = 0.0
		elif np.max(A@sk + gbar) > 0.0:
			idx = np.argmax(A@sk + gbar)
			if idx == 0:
				A[0, :2] = np.array([-2.0, 2.0*x[1]])
			elif idx == 1:
				A[1, :2] = np.array([5.0, 2.0*x[1] - 2.0])

	return sk, mu

if __name__ == "__main__":
	# main script

	# Iteration counter
	k = np.uint16(0)

	# Initial solution guess and step size
	sk = np.zeros((2, 1000), dtype=np.single)
	x = np.zeros((2, 1000), dtype=np.single)
	x[:, :1] = np.array([[1.0], [1.0]])

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

	# Test QP
	[sk[:, k], mu[:, k]] = QP(x[:, k], mu[:, k], W[:, :, k])
