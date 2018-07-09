# this script simulates stage-structured population sizes in space and time for given parameters and forcin
# functions

# NB for EAM: this runs on Mushu in pipenv, so call pipenv run python fish_data_generation.py

# import dependencies
import numpy as np
import random as random
import math as math

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

	nextN_B = neg_pop_reset(nextN_B)
	nextN_J = neg_pop_reset(nextN_J)
	nextN_A = neg_pop_reset(nextN_A)

	return nextN_B, nextN_J, nextN_A


def deterministic_movement(N_Ai, N_Aj, params):
	""" Takes in the population size at patch i and patch j and the fraction that stays in the home patch; 
	returns the population in patch i and patch j after movement """
	next_NAi = params["f_s"]*N_Ai + (1 - params["f_s"])*N_Aj
	next_NAj = params["f_s"]*N_Aj + (1 - params["f_s"])*N_Ai
	return next_NAi, next_NAj


def neg_pop_reset(n):
	"Takes a population size and returns it if it is positive; returns 0 otherwise"
	if n < 0:
		n = 0
	return n


def set_landscape_temps(locations, gradient, offset): 
	""" Takes in a metric of landscape locations and the gradient they should follow and outputs temperatures"""
	temperatures = [x*gradient + offset for x in locations]
	return temperatures


def simulate_population(N0, PARAMS, T_FINAL, LANDSCAPE_LEN):
	LANDSCAPE = []

	for x in range(0,LANDSCAPE_LEN):
		LANDSCAPE.append(x)

	# Generate a landscape of temperatures over time
	temperatures = []
	for t in range(0,T_FINAL):
		landscape_at_t = set_landscape_temps(LANDSCAPE, 1, 0+PARAMS["delta_t"]*t)
		temperatures.append(landscape_at_t)

	#print(temperatures)
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
			PARAMS["alpha"] = growth_rate(temperatures[t][i], PARAMS["T0"], PARAMS["alpha0"], PARAMS["width"])
			#print(PARAMS["alpha"])
			nextNB, nextNJ, nextNA = deterministic_pop_dynamics(N_B[t][i], N_J[t][i], N_A[t][i], PARAMS)
			#print(nextNB, nextNJ, nextNA)
			N_B[t+1][i] = (np.random.poisson(nextNB))
			N_J[t+1][i] = (np.random.poisson(nextNJ))
			N_A[t+1][i] = (np.random.poisson(nextNA))
			#print(N_B[t+1][i], N_J[t+1][i], N_A[t+1][i])
		N_A[t+1][i], N_A[t+1][i] = deterministic_movement(N_A[t+1][i], N_A[t+1][i], PARAMS) # does not generalize 

	return N_B, N_J, N_A


def dispersal_kernel(LANDSCAPE, lambda_val):
	""" Takes in a 1-D landscape vector of length n and a lambda parameter for an expoential distribution
	and returns an nxn matrix where each row is that element's dispersal kernel to the rest of the landscape"""
	kernel = []
	for i in range(0, len(LANDSCAPE)):
		LANDSCAPE = np.array(LANDSCAPE)
		dist = np.abs((LANDSCAPE[i] - LANDSCAPE))
		#print(LANDSCAPE[i], dist)
		exp_element = lambda_val*np.exp(-lambda_val*dist)
		exp_proportion = exp_element / exp_element.sum()
	
		kernel.append(list(exp_proportion))
	return kernel


def calculate_summary_stats(N_B, N_J, N_A):
	"""Takes in a matrix of time x place population sizes for each stage and calculates summary statistics"""
	total_adult = N_A.sum(axis=1) # total population in each stage, summed over space
	total_juv = N_J.sum(axis=1) 
	total_larv = N_B.sum(axis=1) 

	total_population = total_adult + total_juv + total_larv # total population size in each time
	proportion_adult = total_adult / total_population
	proportion_p1 = (N_B[:,0] + N_J[:,0] + N_A[:,0]) / total_population # proportion of population in patch 1

	return total_population, proportion_adult, proportion_p1

# Sets parameters
PARAMS = {"alpha0": 2, "T0": 0, "width": 1, "g_B": .3, "g_J": .4, "m_J": .05, "m_A": .05, "f_s": 1, "delta_t":.1}
T_FINAL = 4
LANDSCAPE_LEN = 5
N0 = 5


print(dispersal_kernel([1,2,3,4,5], 1))

# Simulates population
N_B, N_J, N_A = simulate_population(N0, PARAMS, T_FINAL, LANDSCAPE_LEN)
#print(N_B)
#print(N_J)
#print(N_A)
obs_total_pop, obs_prop_ad, obs_prop_p1 = calculate_summary_stats(N_B, N_J, N_A)
#print(obs_total_pop, obs_prop_ad, obs_prop_p1)

RUN_SIM = False
# Pulls parameters from paramater priors
if RUN_SIM:
	PARAMS_ABC = PARAMS # copies parameters so new values can be generated

	param_save = [[0,0]] # sets an initial 0
	for i in range(0,10000):
		g_J_theta = np.random.beta(2,2)
		alpha0_theta = np.random.lognormal(1,1)

		PARAMS_ABC["g_J"] = g_J_theta # sets the g_J parameter to our random guess
		PARAMS_ABC["alpha0"] = alpha0_theta

		N_B_sim, N_J_sim, N_A_sim = simulate_population(N0, PARAMS_ABC, T_FINAL, LANDSCAPE_LEN) # simulates population with g_J value
		sim_total_pop, sim_prop_ad, sim_prop_p1 = calculate_summary_stats(N_B_sim, N_J_sim, N_A_sim)
		pop_diff = (sim_total_pop - obs_total_pop) / obs_total_pop # percent difference in pop size; will fail at obs = 0
		adult_prop_diff = (sim_prop_ad - obs_prop_ad) / obs_prop_ad 
		pop_p1_diff = (sim_prop_p1 - obs_prop_p1) / obs_prop_p1

		pop_check = all(pop_diff<0.01) # checks if all values of total population within 10% of observed
		ap_check = all(adult_prop_diff<0.01) # checks if all values of adult proportion are within 10% of observed
		p1_check = all(pop_p1_diff<0.01)
		#print(pop_check, ap_check)
		if all([pop_check, ap_check, p1_check]): # if both summary stats are within bounds
			param_save.append([g_J_theta, alpha0_theta]) # saves the parameter value if it was within 10% of observed for all summary stats
			#print(pop_check, ap_check)
	#print(param_save)
	# Makes a model to fit the data
	print(param_save)
#	print(np.histogram(param_save,10))
