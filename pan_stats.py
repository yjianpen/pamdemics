import numpy as np
import matplotlib.pyplot as plt

def get_R0(stats):
	infected=stats[:,4]
	recovered=stats[:,1]
	deaths=stats[:,2]
	susceptible=stats[:,0]
	r0=[]
	for i in range(1,len(stats)):
		if i%7==0:
			beta=(infected[i]-infected[i-7])/susceptible[i-7]
			gamma=((recovered[i]-recovered[i-7])+(deaths[i]-deaths[i-7]))/infected[i-7]
			r0.append(float(beta/(gamma+beta)))
	return np.array(r0)

def plot_R0(R0):
	pass