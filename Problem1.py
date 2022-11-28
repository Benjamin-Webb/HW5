# SQP Algorithm for HW5
# Benjamin Webb
# 11/21/2022

import numpy as np

def objfun(x, k):
	# objective function
	# x: 2x1 vector
	# k: iteration number, int

	return x[0, k]**2 + (x[1, k] - 3.0)**2

def constraints(x, k):
	# vector of inequality constraints
	# x: 2x1 vector
	# k: iteration number, int

	g = np.zeros((2, 1), dtype=np.single)
	g[0] = x[1, k]**2 - 2*x[0, k]
	g[1] = (x[1, k] - 1.0)**2 + 5.0*x[0, k] - 15.0

	return g

def gradLagrangian(x, mu, k):
	# Calculates gradient of the Lagrangian
	# x: 2x1 vector
	# g: 2x1 vector
	# k: interation number, int

	gradL = np.zeros((2, 1), dtype=np.single)
	gradL[0] = 2*x[0, k] - 2*mu[0, k] + 5*mu[1, k]
	gradL[1] = 2*x[1, k] - 6.0 + 2*mu[0, k]*x[1, k] + 2*mu[1, k]*x[1, k] - 2*mu[1, k]

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

if __name__ == "__main__":
	# main script

	# Iteration counter
	k = np.uint16(0)

	# Initial solution guess
	x = np.zeros((2, 1000), dtype=np.single)
	x[:, :1] = np.array([[1.0], [1.0]])

	# Determine if any of the inequality constraints are active
	g = np.zeros((2, 1000), np.single)
	g[:, :1] = constraints(x, k)
	mu = np.zeros((2, 1000), np.single)

	# Calculate gradient of Lagrangian at x0
	gradL = np.zeros((2, 1000), dtype=np.single)
	gradL = gradLagrangian(x, mu, k)
