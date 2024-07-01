import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

N = 30
n_exp = 100_000
ep_length = 60
gamma = 0.99
EPSILON_DECAY = 0.9995

action_dict = {i:v for i,v in enumerate([np.array([1,0]),np.array([0,1]), np.array([-1,0]),np.array([0,-1])])}

def step(pos, a):

	next_pos = pos + action_dict[a]
	next_x , next_y = next_pos

	return np.array([(max(min(N-1, next_x), 0)),(max(min(N-1, next_y), 0))])



SALIENT_POINTS = [np.array([np.random.randint(N),np.random.randint(N)]), np.array([np.random.randint(N),np.random.randint(N)]), np.array([np.random.randint(N),np.random.randint(N)]),  np.array([np.random.randint(N),np.random.randint(N)])]
SALIENT_POINTS = [np.array([1,1]),  np.array([13,5]),  np.array([2,11])]

grid = np.zeros(shape = (N,N))

visitation_grid = np.zeros(shape = (N,N))

OPTIMAL_Q_DICT = {i: np.zeros(shape = (N,N, 4)) for i in range(len(SALIENT_POINTS))}

#print(SALIENT_POINT_DICT)


print(SALIENT_POINTS)

for e in tqdm(range(10_000)):

	s = np.array([np.random.randint(N), np.random.randint(N)])
	
	path = []
	for t in range(ep_length):
		

		a = np.random.randint(4)

		ss = step(s,a)

		path.append((s,a,ss))

		s = ss

	for index, point in enumerate(SALIENT_POINTS):
			
		for (s,a,ss) in path:

			r = 0 
			terminal = 0

			if tuple(point) == tuple(ss):
				r = 1
				terminal = 1
			if tuple(point) == tuple(s):
				terminal = 1

			OPTIMAL_Q_DICT[index][*s][a] = OPTIMAL_Q_DICT[index][*s][a] + (r + (1-terminal)*gamma*np.max(OPTIMAL_Q_DICT[index][*ss]) -OPTIMAL_Q_DICT[index][*s][a])


			


				
			
plt.imshow(np.max(OPTIMAL_Q_DICT[1], axis = 2), cmap='Blues', interpolation='nearest')
# Add colorbar 
print(np.max(OPTIMAL_Q_DICT[1], axis = 2))
plt.colorbar() 
plt.show()



# CREATING THE EVENT VALUE FUNCTION

EVENT_Q_DICT = {i: np.zeros(shape = (N,N, 4)) for i in range(len(SALIENT_POINTS))}
Q_EXPLORER = np.zeros(shape = (N,N, 5))


def get_utility():

	global N, SALIENT_POINTS, EVENT_Q_DICT

	EVENT_VALUE = {i: np.max(EVENT_Q_DICT[i], axis = 2) for i in range(len(SALIENT_POINTS))}

	q = np.zeros(shape = (N,N))
	n = np.zeros(shape = (N,N))

	prob_list = [[[] for _ in  range(N)] for _ in range(N)]

	for idx in range(len(SALIENT_POINTS)):

		for i in range(N):
			for j in range(N):

				if EVENT_VALUE[idx][i][j]>0:
					prob_list[i][j].append(EVENT_VALUE[idx][i][j])

	for i in range(N):
		for j in range(N):
			n[i][j] = len(prob_list[i][j])

			if len(prob_list[i][j]) == 0: 
				q[i][j] = 0
			else:
				q[i][j] = np.min(prob_list[i][j])
			

	U = np.power(2, n-1)*(1 + q)*np.power((1/(gamma**10)),n-1)

	return U


def get_state_utility(s):

	global N, SALIENT_POINTS, EVENT_Q_DICT

	#print(EVENT_Q_DICT[0][*s])
	EVENT_VALUE = {i: np.max(EVENT_Q_DICT[i][*s]) for i in range(len(SALIENT_POINTS))}

	prob_list = []

	for idx in range(len(SALIENT_POINTS)):

		if EVENT_VALUE[idx]>0:
			prob_list.append(EVENT_VALUE[idx])

	n = len(prob_list)
	if len(prob_list)>0:
		q = np.min(prob_list)
	else:
		q = 0

	U = (2 **(n-1))*(1 + q)*((1/(gamma**10))**(n-1))

	return U


episode_completion = {i: np.zeros(n_exp) for i in range(len(SALIENT_POINTS))}
exit_point_list = []
subotimality_value_list = []


for e in tqdm(range(n_exp)):

	s = np.array([int(N/2), int(N/2)])

	path = []

	flag_random_explore = False

	for t in range(ep_length):

		a = None
		if np.random.uniform()<0.1:

			a = np.random.randint(5)
		else:
			max_q = Q_EXPLORER[*s].max()

			possible_actions = []

			for p_a in range(5):
				if Q_EXPLORER[*s][p_a] == max_q:
					possible_actions.append(p_a)

			a = random.choice(possible_actions)

		executed_action = a
		terminal = 0
		r = 0
		if flag_random_explore or a == 4:

			executed_action = np.random.randint(4)
			flag_random_explore = True
			terminal = 1
			exit_point_list.append(s)
			r = get_state_utility(s)

		ss = step(s,executed_action)

		path.append((s,executed_action,ss))

		Q_EXPLORER[*s][executed_action] = Q_EXPLORER[*s][executed_action] + (r + (1-terminal) * gamma * np.max(Q_EXPLORER[*ss]) - Q_EXPLORER[*s][executed_action])


		s = ss

	
	for index, point in enumerate(SALIENT_POINTS):
			
		for (s,a,ss) in path:

			r = 0 
			terminal = 0

			if tuple(point) == tuple(ss):
				r = 1
				terminal = 1
				episode_completion[index][e] = 1
			if tuple(point) == tuple(s):
				terminal = 1

			EVENT_Q_DICT[index][*s][a] = EVENT_Q_DICT[index][*s][a] + (r + (1-terminal)*gamma*np.max(EVENT_Q_DICT[index][*ss]) -EVENT_Q_DICT[index][*s][a])



	EVENT_VALUE = {i: np.max(EVENT_Q_DICT[i], axis = 2) for i in range(len(SALIENT_POINTS))}
	OPTIMAL_EVENT_VALUE  = {i: np.max(OPTIMAL_Q_DICT[i], axis = 2) for i in range(len(SALIENT_POINTS))}

	subotimality_value = sum([(np.nan_to_num(OPTIMAL_EVENT_VALUE[i])-np.nan_to_num(EVENT_VALUE[i])).sum() for i in range(len(SALIENT_POINTS))])
	subotimality_value_list.append(subotimality_value)


U  = get_utility()
plt.imshow(np.max(EVENT_Q_DICT[1], axis = 2), cmap='Blues', interpolation='nearest')
# Add colorbar 
print(np.max(EVENT_Q_DICT[1], axis = 2))
plt.colorbar() 
plt.show()

plt.imshow(U, cmap='Blues', interpolation='nearest')
# Add colorbar 
print(np.max(EVENT_Q_DICT[1], axis = 2))
plt.colorbar() 
plt.show()

#event = episode_completion[0] + episode_completion[1] + episode_completion[2]
ans = {i:np.zeros(n_exp) for i in range(len(SALIENT_POINTS))}


for sl in range(len(SALIENT_POINTS)):

	for i in range(n_exp):
		ans[sl][i] = episode_completion[sl][max(0, i-500):i].mean()

#plt.plot(ans)

#plt.show()

plt.stackplot(range(n_exp),ans[0], ans[1],ans[2], labels=['A','B','C'])
plt.legend(loc='upper left')
plt.show()

exit_points = np.zeros(shape = (N,N))

for i, point in enumerate(exit_point_list[0:1000]):

	exit_points[*point] += 1 
	

plt.imshow(exit_points*2, cmap='Oranges', interpolation='nearest')
# Add colorbar 

plt.colorbar() 
plt.show()


exit_points = np.zeros(shape = (N,N))

for i, point in enumerate(exit_point_list[-1000:]):

	exit_points[*point] += 1 
	

plt.imshow(exit_points*2, cmap='Reds', interpolation='nearest')
# Add colorbar 

plt.colorbar() 
plt.show()


plt.plot(subotimality_value_list)
plt.show()

