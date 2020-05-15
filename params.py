def load_params():
	R0 = 2.4 #@param {type:"slider", min:0.9, max:5, step:0.1}
	#@markdown Disease periods in days
	t_incubation = 5.1 #@param {type:"slider", min:1, max:14, step:0.1}
	t_infective = 3.3 #@param {type:"slider", min:1, max:14, step:0.1}
	#@markdown Population Size
	N = 14000 #@param {type:"slider", min:1000, max:350000, step: 1000}
	alpha = 1/t_incubation
	gamma = 1/t_infective
	beta = R0*gamma
	death_rate = 0.2
	return N,alpha,gamma,beta/3,death_rate


def synthetic_params():
	N=500
	beta=0.03
	alpha=0.032
	gamma=0.05
	return N,alpha,gamma,beta,death_rate