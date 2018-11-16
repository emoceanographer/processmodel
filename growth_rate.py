# define custom functions
def growth_rate(temperature, T0, alpha0, width):
	""" Takes in a temperature as well as the location of the parabolic vertex (T0, alpha0) and width of
	the parabola and returns the associated growth rates 
	"""
	alpha = -width*(temperature-T0)*(temperature-T0) + alpha0
	return alpha	