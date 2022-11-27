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

if __name__ == '__SQP__':
	# Main script

	# Initial solution guess
	x = np.array([[1.0], [1.0]], dtype=np.single)

	# Determine if any of the inequality constraints are active
	g = constraints(x)
