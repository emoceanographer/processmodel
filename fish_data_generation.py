# this script simulates stage-structured population sizes in space and time for given parameters and forcin
# functions

# NB for EAM: this runs on Mushu in pipenv, so call pipenv run python fish_data_generation.py

# import dependencies
import numpy as np
import random as random
import math as math

import growth_rate as growth_rate


def temp_dependence(temperature, T0, theta_0, width):
	""" Takes in a temperature and the location of the parabolic vertex (T0, theta_0) and the width of
	the parabola and returns the associated temperature dependent rate (theta)
	"""
	theta = -width*(temperature-T0)*(temperature-T0) + theta_0
	return theta


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

	kernel = dispersal_kernel(LANDSCAPE, PARAMS["lam"])
	kernel = np.array(kernel) # having this as an array makes calls later (columsn alone) much easier
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
			#print(type(N_A), type(kernel))
		N_Am = np.ones(LANDSCAPE_LEN) - 1
		#print(N_Am)
		for i in range(0, LANDSCAPE_LEN): # runs through each element again for dispersal
			#print(N_A[t+1,:])
			#print(kernel[:,i])
			#print(N_A[t+1,:]*kernel[:,i])
			incoming_fish = (1-PARAMS["f_s"])*N_A[t+1,:]*kernel[:,i] # contribution of all other patches to patch i
			N_Am[i] = N_A[t+1][i]*PARAMS["f_s"] +  incoming_fish.sum() # add "fraction stay" plus the fraction leaving
			#from other patches

		N_A[t+1,:] = N_Am
	

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
	proportion_p_i = []

	for i in range(0, len(N_B)):
		proportion_p_i.append((N_B[i,0]+N_J[i,0]+N_A[i,0])/total_population[i])

	proportion_p_i = np.asarray(proportion_p_i)
	return total_population, proportion_adult, proportion_p_i

# Sets parameters
PARAMS = {"alpha0": 2, "T0": 0, "width": 1, "g_B": .3, "g_J": .4, "m_J": .05, "m_A": .05, "f_s": .9, "delta_t":.1, "lam":1}
T_FINAL = 10
LANDSCAPE_LEN = 2
N0 = 5
NUMBER_SIMS = 100000

print("True parameter values:", PARAMS["g_J"], PARAMS["alpha0"], PARAMS["width"], PARAMS["f_s"], PARAMS["lam"])

# Simulates population
N_B, N_J, N_A = simulate_population(N0, PARAMS, T_FINAL, LANDSCAPE_LEN)

obs_total_pop, obs_prop_ad, obs_prop_p_i = calculate_summary_stats(N_B, N_J, N_A)


RUN_SIM = True
# Pulls parameters from paramater priors
if RUN_SIM:
	PARAMS_ABC = PARAMS # copies parameters so new values can be generated; FIX ME! this is a redirect, not a copy?

	param_save = [] # sets an initial 0; fixed to [] because [[]] made the mean go poorly (averaging in an [] at start?)
	for i in range(0,NUMBER_SIMS):
		g_J_theta = np.random.beta(2,2)
		alpha0_theta = np.random.lognormal(1,1)
		width_theta = np.random.lognormal(1,1)
		f_s_theta = np.random.uniform()
		lam_theta = np.random.lognormal(1,1)

		PARAMS_ABC["g_J"] = g_J_theta # sets the g_J parameter to our random guess
		PARAMS_ABC["alpha0"] = alpha0_theta
		PARAMS_ABC["width"] = width_theta
		PARAMS_ABC["f_s"] = f_s_theta
		PARAMS_ABC["lam"] = lam_theta

		N_B_sim, N_J_sim, N_A_sim = simulate_population(N0, PARAMS_ABC, T_FINAL, LANDSCAPE_LEN) # simulates population with g_J value
		sim_total_pop, sim_prop_ad, sim_prop_p_i = calculate_summary_stats(N_B_sim, N_J_sim, N_A_sim)
		pop_diff = abs((sim_total_pop - obs_total_pop) / obs_total_pop) # percent difference in pop size; will fail at obs = 0
		adult_prop_diff = abs((sim_prop_ad - obs_prop_ad)) 
		pop_p1_diff = abs((sim_prop_p_i - obs_prop_p_i))


		pop_check = all(pop_diff<0.3) # checks if all values of total population within % of observed
		ap_check = all(adult_prop_diff<0.1) # checks if all values of adult proportion are within % of observed
		p1_check = np.all(pop_p1_diff<0.4) # np.all needed to resolve an ambiguous all call on an array



		if all([pop_check, ap_check, p1_check]): # if both summary stats are within bounds
			param_save.append([g_J_theta, alpha0_theta, width_theta, f_s_theta, lam_theta]) # saves the parameter value if it was within 10% of observed for all summary stats

	# Makes a model to fit the data
	param_save = np.array(param_save) # added this because without it, I get an error by taking the mean (array function)
	
	print("Total number of acceptable parameter runs", len(param_save), "out of", NUMBER_SIMS)
	mean_params = np.mean(param_save, axis=0)
	std_params = np.std(param_save, axis = 0)
	print("Mean of parameters:", mean_params)
	print("StDev of parameters:", std_params)


