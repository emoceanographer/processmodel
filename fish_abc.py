# this script simulates stage-structured population sizes in space and time for given parameters and forcin
# functions

# NB for EAM: this runs on Mushu in pipenv, so call pipenv run python fish_abc.py

# import dependencies
import numpy as np
import random as random
import math as math
import copy as copy
import itertools

def beta_defs(alpha,mu):
	""" Takes in an alpha shape parameter and the desired mean and returns a value from the beta
	distribution"""
	beta = alpha/mu - alpha
	val = np.random.beta(alpha, beta)
	return val


def temp_dependence(temperature, T0, theta_0, width):
	""" Takes in a temperature and the location of the parabolic vertex (T0, theta_0) and the width of
	the parabola and returns the associated temperature dependent rate (theta)
	"""
	theta = -width*(temperature-T0)*(temperature-T0) + theta_0
	if theta < 0:
		theta = 0

	return theta


def pop_dynamics(N_B, N_J, N_A, params, alpha, alph):
	""" Intakes the numeric population sizes for the three population sizes and a dict structure
	for the parameter values; alpha, fecundity is input separately
	"""
	recruits = beta_defs(alph,params["g_B"])
	gJval = beta_defs(alph,params["g_J"])
	nJ_coef = (1 - gJval - beta_defs(alph,params["m_J"]))

	nextN_B = np.random.poisson(alpha)*N_A - recruits*N_B
	nextN_J = recruits*N_B + N_J*nJ_coef
	nextN_A = gJval*N_J + (1 - beta_defs(alph,params["m_A"]))*N_A

	return nextN_B, nextN_J, nextN_A

def movement(pop1, pop2, f_s):
	""" Takes population sizes in two patches, in which a fraction, f_s, of each stays and outputs
	the population sizes in each patch after movement """
	next_pop1 = f_s*(pop1) + (1-f_s)*pop2
	next_pop2 = f_s*(pop2) + (1-f_s)*pop1

	return next_pop1, next_pop2


def simulation_population(N_B0, N_J0, N_A0, params, T_FINAL, temperatures):
	""" Takes in the initial population sizes and simulates the population size moving forward """
	# Set starting numbers for population and allocate space for population sizes
	N_B = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
	N_B[0] = N0
	N_J = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
	N_J[0] = N0
	N_A = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
	N_A[0] = N0
	for t in range(0,T_FINAL):
		alpha1 = temp_dependence(temperatures[t], params["T0"], params["alpha0"], params["width"])
		alpha2 = temp_dependence(temperatures[t] + params["delta_t"], params["T0"], params["alpha0"], params["width"])

		N_B[t+1][0], N_J[t+1][0], N_A1 = pop_dynamics(N_B[t][0], N_J[t][0], N_A[t][0], params, alpha1, 2)
		N_B[t+1][1], N_J[t+1][1], N_A2 = pop_dynamics(N_B[t][1], N_J[t][1], N_A[t][1], params, alpha2, 2)

		N_A[t+1][0], N_A[t+1][1] = movement(N_A1,N_A2, params["f_s"])

	return N_B, N_J, N_A


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


def euclidean_distance(vec1, vec2):
	""" Takes two vectors of the same dimensions and calculates the Euclidean distance between the elements"""
	d = 0
	if len(vec1) != len(vec2):
		print("Error")
		return

	for i in range(0,len(vec1)):
		d = d + (vec1[i] - vec2[i])**2
	distance = d**0.5
	return distance


def small_percent(vector, percent):
	""" Takes a vector and returns the indexes of the elements within the smallest (percent) percent of the vector"""
	sorted_vector = sorted(vector)
	cutoff = math.floor(len(vector)*percent/100) # finds the value which (percent) percent are below
	indexes = []
	print(cutoff)
	for i in range(0,len(vector)):
		if vector[i] <= sorted_vector[cutoff]: # looks for values below the found cutoff
			indexes.append(i)

	return indexes


# Sets parameters
PARAMS = {"alpha0": 2, "T0": 0, "width": 1, "g_B": .3, "g_J": .4, "m_J": .05, "m_A": .05, "f_s": .9, "delta_t":.1, "lam":1}
T_FINAL = 10
LANDSCAPE_LEN = 2
N0 = 5
NUMBER_SIMS = 100

print("True parameter values:", PARAMS["g_J"], PARAMS["g_B"],PARAMS["alpha0"], PARAMS["width"], PARAMS["f_s"], PARAMS["T0"], PARAMS["m_J"], PARAMS["m_A"])

# Sets temperatures at each patch over time [FIX THIS]
temperatures = [0,1,2,3,4,5,6,7,8,9]


# Simulates population
N_B, N_J, N_A = simulation_population(5,5,5,PARAMS,T_FINAL, temperatures)

obs_total_pop, obs_prop_ad, obs_prop_p_i= calculate_summary_stats(N_B, N_J, N_A)

RUN_SIM = True
# Pulls parameters from paramater priors
if RUN_SIM:
    PARAMS_ABC = copy.deepcopy(PARAMS) # copies parameters so new values can be generated; FIX ME! this is a redirect, not a copy?

    param_save = [] # sets an initial 0; fixed to [] because [[]] made the mean go poorly (averaging in an [] at start?)
    dists = []
    for i in range(0,NUMBER_SIMS):
        g_B_theta = np.random.beta(2,2)
        g_J_theta = np.random.beta(2,2)
        alpha0_theta = np.random.lognormal(1,1)
        width_theta = np.random.lognormal(1,1)
        f_s_theta = np.random.uniform()
        T0_theta = np.random.normal(0,0.5)
        m_J_theta = np.random.beta(2,2)
        m_A_theta = np.random.beta(2,2)

        PARAMS_ABC["g_J"] = g_J_theta # sets the g_J parameter to our random guess
        PARAMS_ABC["alpha0"] = alpha0_theta
        PARAMS_ABC["width"] = width_theta
        PARAMS_ABC["f_s"] = f_s_theta
        PARAMS_ABC["g_G"] = g_B_theta
        PARAMS_ABC["T0"] = T0_theta
        PARAMS_ABC["m_J"] = m_J_theta
        PARAMS_ABC["m_A"] = m_A_theta

        N_B_sim, N_J_sim, N_A_sim = simulation_population(N0,N0,N0, PARAMS_ABC, T_FINAL, temperatures) # simulates population with g_J value
        #sim_total_pop, sim_prop_ad, sim_prop_p_i = calculate_summary_stats(N_B_sim, N_J_sim, N_A_sim)
        
        #vec1 = sim_total_pop + sim_prop_ad + sim_prop_p_i
        #vec2 = obs_total_pop + obs_prop_ad + obs_prop_p_i
        a = list(itertools.chain.from_iterable(N_B_sim)) # I had added these in lieu of the summary stats as the 'real'
        #data that we are trying to replicate mostly closely
        b = list(itertools.chain.from_iterable(N_J_sim))
        c = list(itertools.chain.from_iterable(N_A_sim))
        d = list(itertools.chain.from_iterable(N_B))
        e = list(itertools.chain.from_iterable(N_J))
        f = list(itertools.chain.from_iterable(N_A))
        vec1 = a + b + c
        vec2 = d + e + f

        param_save.append([g_J_theta, g_B_theta, alpha0_theta, width_theta, f_s_theta, T0_theta, m_J_theta, m_A_theta])
        dists.append(euclidean_distance(vec1,vec2))

    library_index = small_percent(dists,5)
    library = []
    for i in range(0,len(library_index)):
        library.append(param_save[library_index[i]])
    #print(library)
