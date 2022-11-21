# SQP Algorithm for HW5
# Benjamin Webb
# 11/21/2022

import numpy as np

def objfun(x):
	# objective function

	return x[0]**2 + (x[1] - 3.0)**2
