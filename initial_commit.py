import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
position_likelihood_list=dict()
position_likelihood_list["student"]={"school":0.3,"shop":0.3,"entertainment":0.4,"hospital":0,"home":1}
position_likelihood_list["doctor"]={"school":0,"shop":0.3,"entertainment":0.4,"hospital":0.9,"home":1}
position_likelihood_list["clerk"]={"school":0,"shop":0.3,"entertainment":0.4,"hospital":0,"home":1}
position_likelihood_list["esssential workers"]={"school":0,"shop":1.0,"entertainment":0.4,"hospital":0,"home":1}

class person:
	def __init__(self,coord,career,id,age=40):
		self.coord=coord
		self.vehicle_status=None
		self.age=age
		self.career=career
		self.status="susceptible"
		self.id=id
		self.home=coord #Assume that home is their initial coordinate
		self.dailyhistory=[]
		self.position_likelihood={"school":0.3,"shop":0.3,"entertainment":0.4,"home":1}## Let's try homogenous career first
		self.position_timeperiod=[]
		self.exposure_date=-1

	def move(self,destination,mode,speed=[0,0]):
		# Process parameters
		delta = 0.25
		dt = 0.1
		# Initial condition.
		x = 0.0
		# Iterate to compute the steps of the Brownian motion.
		for k in range(n):    
			self.coord=destination
			if mode=="local":
				self.coord[0]+=norm.rvs(scale=delta**2*dt)
				self.coord[1]+=norm.rvs(scale=delta**2*dt)
			else:
				self.coord[0]+=speed[0]
				self.coord[1]+=speed[1]


		






class map:
	def __init__(self,row,col,career):
		self.row=row
		self.col=col
		self.buildings=[]
		self.building_map=np.zeros([row,col])
		self.human_map=np.zeros([row,col])
		self.people=dict()
		self.career=career
		self.traffic=[]
		self.total_infected=0
		self.total_susceptible=len(self.people.keys())
		self.total_death=0
		self.total_recovered=0
		self.total_exposed=0
		self.date=0



	def add_buildings(self,buildings):
		self.buildings=buildings

	##home is a special type of building
	def add_home(self,people):
		home_addresses=[]
		for person in people:
			if person.home not in home_addresses:
				home_addresses.append(person.home)
				new_home=building(typeb='home',coord=person.home,id="h"+str(len(home_addresses)-1))
				self.buildings.append(new_home)
			else:
				pass
				'''
				print("repeat!")
				'''
	def population_density():
		return np.sum(human_map)/(len(human_map)*len(human_map[0]))

	def total_infected():
		infected=0
		for person in people:
			if person.status=="infected":
				infected+=1
		return infected

	def get_distance(x,y):
		return math.abs(x[0]-y[0])+math.abs(x[1]-y[1])

	def add_people(self,num_people,x=-1,y=-1,career='',family_size=3):
		for i in range(num_people):
			if x==-1:
				new_x=np.random.randint(self.row)
			elif num_people%family_size==0:
				new_x=x
			if y==-1:
				new_y=np.random.randint(self.col)
			elif num_people%family_size==0:
				new_y=y
			if career=='':
				new_career=self.career[np.random.randint(len(self.career))]
			new_coord=[new_x,new_y]
			new_person=person(new_coord,new_career,i)
			self.people[i]=new_person
		self.total_susceptible=num_people
	def print_people(self):
		for key in self.people:
			print("id",self.people[key].id,"career",self.people[key].career,"coord",self.people[key].coord)

	def shelter_policy(self,new_likelihood):
		for person in self.people.values():
			person.position_likelihood=new_likelihood

	def plt_map(self,grid_mode=True):
		r=self.row
		c=self.col
		y=np.linspace(0, c, c+1, endpoint=True)
		if grid_mode:
			for i in y:
				x1 = np.linspace(0, r, r+1, endpoint=True)
				y2 = np.array([i]*(r+1))
				plt.plot(x1, y2, 'o',color='black')
		if self.traffic==[]:
			pass
		else:
			print("yes!")
			for edge in self.traffic:
				point1=edge[0]
				point2=edge[1]
				x_values = [point1[0], point2[0]]
				y_values = [point1[1], point2[1]]
				plt.plot(x_values, y_values,color='black')
		plt.show()

	def add_traffic(self,edges): #Traffic will be addeed as a list of edges
		self.traffic=edges
		for edge in edges:
			if np.array(edge[0]).any()<0 or np.array(edge[1]).any()<0:
				raise ValueError("Node number smaller than 0!")
			if np.array(edge[0]).any()>self.row or np.array(edge[1]).any()>self.col:
				raise ValueError("Node number greater than maximal dimension!")

	def plt_people(self):
		for key in self.people:
			p=self.people[key]
			x,y=p.coord[0],p.coord[1]
			plt.plot(x,y,'o',color='red')
		plt.show()

	def get_all_people(self):
		return self.people.keys()


	def local_movement(self,person,E_mtx):
		if self.get_distance(person.coord-person.home)>(len(E_mtx)-1):
			pass
		else:
			center=([(len(E_mtx)+1)/2],[(len(E_mtx)+1)/2])
			center[0]+=(self.coord[0]-self.home[0])
			center[1]+=(self.coord[1]-self.home[1])
	
	def get_closest_builing(self,typeb,person): ## Working on that function
		min_val=10000
		min_b=self.buildings[0]
		for building in self.buildings:
			vec=np.array(person.coord)-np.array(building.coord)
			if building.type==typeb and math.sqrt(np.dot(vec,vec))<min_val:
				min_b=building
				min_val=np.dot(vec,vec)
		return min_b





class building:
	def __init__(self,typeb,coord,id):
		self.type=typeb
		self.coord=coord
		self.id=id
		self.visitors=[]
		self.infection_prob=0.005
		self.death_prob=0.007
		self.detect_prob=0.05
		self.recover_prob=0.025
	def check_infected_visitors(self):
		for person in self.visitors:
			if person.status=="infected":
				return True
		return False

	def update(self,map):
		'''
		print(self.check_infected_visitors())
		if self.check_infected_visitors()==False:
			pass
		else:
		'''
		infection_prob=self.infection_prob
		for person in map.people.values():
			if person.status=="susceptible" and self.check_infected_visitors():
				i=np.random.binomial(1,infection_prob, size=1)
				if i==1:
					person.status="exposed"
					map.total_exposed+=1
					map.total_susceptible-=1
			if person.status=="exposed":
				i=np.random.binomial(1,self.detect_prob, size=1)
				if i==1:
					person.status="infected"
					map.total_exposed-=1
					map.total_infected+=1

			if person.status=="infected":
				i=np.random.binomial(1,self.death_prob, size=1)
				if i==1:
					person.status="dead"
					map.total_infected-=1
					map.total_death+=1
				j=np.random.binomial(1,self.recover_prob, size=1)
				if j==1:
					person.status="recovered"
					map.total_infected-=1
					map.total_recovered+=1
def plot_stats_smooth(stats):
	x=range(len(stats))
	y0 = gaussian_filter1d(stats[:,0], sigma=1)
	y1 = gaussian_filter1d(stats[:,1], sigma=1)
	y2 = gaussian_filter1d(stats[:,2], sigma=1)
	y3 = gaussian_filter1d(stats[:,3], sigma=1)
	p0,=plt.plot(x,y0, 'r')
	p1,=plt.plot(x,y1,'b')
	p2,=plt.plot(x,y2,'g')
	p3,=plt.plot(x,y3,'m')
	plt.legend([p0,p1,p2, p3], ['current infected', 'total recovered','total death','total susceptible'], loc='best', scatterpoints=1)
	plt.xlabel("Days")
	plt.ylabel("Number of People")
	plt.show()

def plot_stats(stats):
	x=range(len(stats))
	y0 = stats[:,0]
	y1 = stats[:,1]
	y2 = stats[:,2]
	y3 = stats[:,3]
	print("y0",y0,"stats0",stats[:,0])

	p0,=plt.plot(x,y0, 'r')
	p1,=plt.plot(x,y1,'b')
	p2,=plt.plot(x,y2,'g')
	p3,=plt.plot(x,y3,'m')
	plt.legend([p0,p1,p2, p3], ['total infected', 'total recovered','total death','total susceptible'], loc='best', scatterpoints=1)
	plt.xlabel("Days")
	plt.ylabel("Number of People")
	plt.show()

def simulate_one_day(map):
	avg_activity=3 ## Average times that a person go outdoor per week
	city_dailyhistory=[]
	for people in map.people.values():
		locations=[]
		likelihood=list(people.position_likelihood.values())
		people.dailyhistory=[np.random.binomial(avg_activity,l, size=1)[0] for l in likelihood]
		city_dailyhistory.append([people.id,people.dailyhistory])
		for i in range(len(people.position_likelihood.values())):
			if people.dailyhistory[i]!=0:
				min_b=map.get_closest_builing(list(people.position_likelihood.keys())[i],people)
				locations.append(min_b)
				min_b.visitors.append(people)
	for building in map.buildings:
		building.update(map)
		building.visitors=[]
	map.date+=1
	print("total_infected:",map.total_infected,"total_recovered",map.total_recovered,"total_death",map.total_death,"total_susceptible",map.total_susceptible,"total_exposed",map.total_exposed)

	return [map.total_infected,map.total_recovered,map.total_death,map.total_susceptible]



def simulate():
	career=['doctor','student','clerk','essential workers']
	map1=map(40,40,career)
	## Initializing some buildings on our map
	hospital1=building(typeb='hospital',coord=[2,2],id=0)
	hospital2=building(typeb='hospital',coord=[8,7],id=1)
	hospital3=building(typeb='hospital',coord=[3,1],id=2)
	school1=building(typeb='school',coord=[7,6],id=3)
	shop1=building(typeb='shop',coord=[6,5],id=4)
	map1.add_people(500)
	## Initializing patient 0
	map1.people[0].status="infected"
	map1.total_infected+=1
	map1.total_susceptible-=1
	map1.add_home(map1.people.values())
	map1.add_buildings([hospital1,hospital2,hospital3,school1,shop1])
	edges=[([1,2],[2,3]),([7,8],[9,10]),([10,1],[9,10])]
	map1.add_traffic(edges)
	'''
	map1.plt_map(grid_mode=True)
	map1.print_people()
	map1.plt_people()
	'''
	stats=[1,0,0,max(map1.people.keys())]
	print("initial stats",stats)
	while map1.date!=55 and map1.total_infected>0:
		if map1.date==3:
			map1.shelter_policy({"school":0.01,"shop":0.01,"entertainment":0.0,"home":1})

		print('\n'+"Day "+str(map1.date)+" statistics:")
		if stats==[]:
			stats=simulate_one_day(map1)
		else:
			stats= np.vstack((stats,simulate_one_day(map1)))
		
	plot_stats(stats)
if __name__ == "__main__":
	simulate()
	'''
	map1.print_people()
	'''







