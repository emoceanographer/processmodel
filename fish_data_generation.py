# this script simulates stage-structured population sizes in space and time for given parameters and forcin
# functions

# NB for EAM: this runs on Mushu in pipenv, so call pipenv run python fish_data_generation.py

# import dependencies
import numpy as np
import random as random

# define custom functions
def growth_rate(temperature, T0, alpha0, width):
	""" Takes in a temperature as well as the location of the parabolic vertex (T0, alpha0) and width of
	the parabola and returns the associated growth rates 
	"""
	alpha = -width*(temperature-T0)*(temperature-T0) + alpha0
	return alpha	


def deterministic_pop_dynamics(N_B, N_J, N_A, params):
	""" Intakes the numeric population sizes for the three population sizes and a dict structure
	for the parameter values
	"""
	nextN_B = params["alpha"]*N_A - params["g_B"]*N_B
	nextN_J = params["g_B"]*N_B + N_J*(1-params["g_J"] - params["m_J"])
	nextN_A = params["g_J"]*N_J + (1 - params["m_A"])*N_A
	return nextN_B, nextN_J, nextN_A


def deterministic_movement(N_Ai, N_Aj, params):
	""" Takes in the population size at patch i and patch j and the fraction that stays in the home patch; 
	returns the population in patch i and patch j after movement """
	next_NAi = params["f_s"]*N_Ai + (1 - params["f_s"])*N_Aj
	next_NAj = params["f_s"]*N_Aj + (1 - params["f_s"])*N_Ai
	return next_NAi, next_NAj


def set_landscape_temps(locations, gradient, offset): 
	""" Takes in a metric of landscape locations and the gradient they should follow and outputs temperatures"""
	temperatures = [x*gradient + offset for x in locations]
	return temperatures


def simulate_population(N0, PARAMS, T_FINAL, LANDSCAPE_LEN):
	LANDSCAPE = []

	for x in range(0,LANDSCAPE_LEN):
		LANDSCAPE.append(x)

	# Generate a landscape of temperatures
	temperatures = set_landscape_temps(LANDSCAPE, 1, 0)


	# Set starting numbers for population and allocate space for population sizes
	N_B = np.ndarray(shape=(T_FINAL+1, LANDSCAPE_LEN), dtype=float, order='F')
	N_B[0] = N0
	N_J = np.ndarray(shape=(T_FINAL+1, LANDSCAPE_LEN), dtype=float, order='F')
	N_J[0] = N0
	N_A = np.ndarray(shape=(T_FINAL+1, LANDSCAPE_LEN), dtype=float, order='F')
	N_A[0] = N0


	# Simulation population forward through time with Poisson 
	for t in range(0, T_FINAL):
		for i in range(0, LANDSCAPE_LEN):
			PARAMS["alpha"] = growth_rate(temperatures[i], PARAMS["T0"], PARAMS["alpha0"], PARAMS["width"])
			#print(PARAMS["alpha"])
			nextNB, nextNJ, nextNA = deterministic_pop_dynamics(N_B[t][i], N_J[t][i], N_A[t][i], PARAMS)
			#print(nextNB, nextNJ, nextNA)
			N_B[t+1][i] = (np.random.poisson(nextNB))
			N_J[t+1][i] = (np.random.poisson(nextNJ))
			N_A[t+1][i] = (np.random.poisson(nextNA))
			#print(N_B[t+1][i], N_J[t+1][i], N_A[t+1][i])
		N_A[t+1][i], N_A[t+1][i] = deterministic_movement(N_A[t+1][i], N_A[t+1][i], PARAMS) # does not generalize 

	return N_B, N_J, N_A

# Sets parameters
PARAMS = {"alpha0": 2, "T0": 0, "width": 1, "g_B": .3, "g_J": .4, "m_J": .05, "m_A": .05, "f_s": 1}
T_FINAL = 2
LANDSCAPE_LEN = 1
N0 = 5

# Simulates population

N_B, N_J, N_A = simulate_population(N0, PARAMS, T_FINAL, LANDSCAPE_LEN)

print(N_B)
print(N_J)
print(N_A)
# Generates summary statistics

# Makes a model to fit the data


