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

def meritfun(mu, ww, k):
	# weights for merit function used in linesearch
	# x: 2x1 vector
	# mu: 2x1 vector containing lagrange multipliers
	# w: 2x2 matrix containing weights for merit function
	# k: iteration number

	if k == 0:
		w = np.abs(mu)
	else:
		w = np.maximum(np.abs(mu), 0.5*(ww + np.abs(mu)))

	return w

def QP(x, mu, W, k):
	# Solves QP subproblem w/ active set
	# x: 2x1 vector
	# mu: 2x1 vector

	# Formulate intial set of active constraints
	# if k == 0:
	# 	A = gradConstraints(x)
	# 	gbar = constraints(x)
	# else:
	A = gradConstraints(x)
	gbar = constraints(x)
	if mu[0] <= 0.0:
		A[0, 0] = 0.0
		A[0, 1] = 0.0
		gbar[0] = 0.0
	if mu[1] <= 0.0:
		A[1, 0] = 0.0
		A[1, 1] = 0.0
		gbar[1] = 0.0

	# Jacobian of objective function
	fx = np.array([[2*x[0, 0]], [2*x[1, 0] - 6.0]], dtype=np.single)
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
		dgdx1 = dg[0:1, 0:2]@sk + gbar[0]
		dgdx2 = dg[1:2, 0:2]@sk + gbar[1]

		# Determine if QP subproblem is solved
		if mu[0] > 0:
			if dgdx1 <= 0.0:
				if mu[1] <= 0.0 and dgdx2 <= 0.0:
					if mu[1] < 0.0:
						mu[1] = 0.0
					break
				elif mu[1] > 0 and dgdx2 <= 0.0:
					break
		if mu[1] > 0:
			if dgdx2 <= 0.0:
				if mu[0] <= 0 or dgdx1 <= 0.0:
					if mu[0] < 0.0:
						mu[0] = 0.0
					break
				elif mu[0] < 0:
					mu[0] = 0.0

		# Update active constraints
		# A = gradConstraints(x)
		# gbar = constraints(x)
		if mu[0] <= 0.0 and mu[0] < mu[1]:
			A[0, 0] = 0.0
			A[0, 1] = 0.0
			gbar[0] = 0.0
		if mu[1] <= 0.0 and mu[1] < mu[0]:
			A[1, 0] = 0.0
			A[1, 1] = 0.0
			gbar[1] = 0.0
		if dgdx1 > 0.0 and dgdx1 > dgdx2:
			A[0, 0] = -2.0
			A[0, 1] = 2*x[1]
			gbar[0] = x[1]**2 - 2*x[0]
		if dgdx2 > 0.0 and dgdx2 > dgdx1:
			A[1, 0] = 5.0
			A[1, 1] = 2*x[1] - 2.0
			gbar[1] = (x[1] - 1.0)**2 + 5*x[0] - 15.0

		j += 1

	if mu[0] < 0.0:
		mu[0] = 0.0
	if mu[1] < 0.0:
		mu[1] = 0.0

	return sk, mu

def linesearch(x, sk, mu, ww, k):
	# Does linesearch for this SQP problem
	# Many variables needed are calculated inside function
	# x: 2x1 vector of current location
	# sk: 2x1 vector of current step size
	# mu: 2x1 vector of current guess of Lagrange multipliers
	# ww: 2x2 matrix containing the merit function weights

	# Intialize linesearch parameters
	t = np.single(1)
	alpha = np.single(1)

	# Calculate weights for merit function
	if k == 0:
		ww = meritfun(mu, ww, k)
	else:
		ww = meritfun(mu, ww, k)

	# Caclulate F(x+a*sk, mu)
	fx = np.array([[2*x[0, 0]], [2*x[1, 0] - 6.0]], dtype=np.single)
	fxs = fx.T @ sk
	g = constraints(x + alpha*sk)
	F = fxs + np.sum(ww * np.maximum(np.array([[0.0], [0.0]]), g))

	# Calculate Phi(alpha)
	g = constraints(x)
	fphi = objfun(x)
	Fx = fphi + np.sum(ww * np.maximum(np.array([[0.0], [0.0]]), g))
	dg = np.array([[-2.0, 2*x[1, 0]], [5.0, 2*x[1, 0] - 2.0]], dtype=np.single) @ sk
	dgdalpha = np.maximum(np.array([[0.0], [0.0]]), dg)
	Phi = Fx + t*alpha*(fxs + np.sum(ww * dgdalpha))

	while F > Phi:
		alpha = alpha / 2

		# Caclulate F(x+a*sk, mu)
		g = constraints(x + alpha * sk)
		F = fxs + np.sum(ww * np.maximum(np.array([[0.0], [0.0]]), g))

		# Calculate Phi(alpha)
		Phi = Fx + t * alpha * (fxs + np.sum(ww * dgdalpha))

		if alpha < 0.00001:
			break

	return alpha, ww

def BFGS(W, alphask, x, mu, k):
	# Function performs BFGS for given optimization problem
	# W: 2x2 array containing previous iteration Hessian approximation
	# alphask: 2x1 vector containing alphak*sk
	# x: 2xn array containing previous iteration solutions
	# mu: 2x1 vector containing new guesses for Lagrange multipliers
	# k: integer for current iteration

	# Determine theta at current iteration
	dLk1 = gradLagrangian(x[:, k+1:k+2], mu)
	dLk0 = gradLagrangian(x[:, k:k+1], mu)

	if alphask.T @ (dLk1 - dLk0) >= 0.2:
		theta = np.single(1)
	else:
		theta = (0.8 * alphask.T @ W @ alphask) / ((alphask.T @ W @ alphask) -
		                                           (alphask.T @ (dLk1 - dLk0)))

	# Calculate y for BFGS
	y = theta*(dLk1 - dLk0) + (1 - theta)*W@alphask

	# Update Hessian approximation
	H = W + (y@y.T) - ((W @ alphask @ alphask.T @ W) / (alphask.T @ W @ alphask))

	return H

if __name__ == "__main__":
	# main script

	# Iteration counter
	k = np.uint16(0)
	eps = 0.001

	# Initial solution guess and step size
	alpha = np.zeros((1000, 1), dtype=np.single)
	sk = np.zeros((2, 1000), dtype=np.single)
	x = np.zeros((2, 1000), dtype=np.single)
	x[:, :1] = np.array([[1.0], [1.0]])

	# Determine if any of the inequality constraints are active
	g = np.zeros((2, 1000), np.single)
	g[:, :1] = constraints(x[:, :1])
	mu = np.zeros((2, 1000), np.single)

	# Calculate gradient of Lagrangian at x0
	gradL = np.zeros((2, 1000), dtype=np.single)
	gradL[:, :1] = gradLagrangian(x[:, :1], mu[:, :1])

	# Initialize W
	W = np.zeros((2, 2, 1000), dtype=np.single)
	W[0, 0, 0] = 1.0
	W[1, 1, 0] = 1.0

	# Initialize array of weights to be used in linesearch
	ww = np.zeros((2, 1000), dtype=np.single)

	# Do SQP loop
	res = np.single(1)
	while res > eps:
		# Run QP
		[sk[:, k:k+1], mu[:, k+1:k+2]] = QP(x[:, k:k+1], mu[:, k:k+1], W[:, :, k], k)

		# Test linesearch
		[alpha[k], ww[:, k+1:k+2]] = linesearch(x[:, k:k+1], sk[:, k:k+1], mu[:, k+1:k+2], ww[:, k:k+1], k)

		# Update solution
		x[:, k+1:k+2] = x[:, k:k+1] + alpha[k]*sk[:, k:k+1]

		# Update Hessian approximation
		W[:, :, k+1] = BFGS(W[:, :, k], alpha[k]*sk[:, k:k+1], x, mu[:, k+1:k+2], k)

		# Update gradient of Lagrangian
		gradL[:, k+1:k+2] = gradLagrangian(x[:, k+1:k+2], mu[:, k+1:k+2])
		k += 1

		res = np.linalg.norm(gradL[:, k] - gradL[:, k-1])
