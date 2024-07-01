import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

N = 50
n_exp = 100_000
ep_length = 160

def step(pos):

	next_pos = pos + random.choice([np.array([1,0]),np.array([0,1]), np.array([-1,0]),np.array([0,-1])])
	next_x , next_y = next_pos

	return np.array([(max(min(N-1, next_x), 0)),(max(min(N-1, next_y), 0))])



SALIENT_POINTS = [np.array([np.random.randint(N),np.random.randint(N)]), np.array([np.random.randint(N),np.random.randint(N)]), np.array([np.random.randint(N),np.random.randint(N)]),  np.array([np.random.randint(N),np.random.randint(N)])]
SALIENT_POINTS = [np.array([1,1]),  np.array([13,5]),  np.array([2,11])]

grid = np.zeros(shape = (N,N))

visitation_grid = np.zeros(shape = (N,N))

SALIENT_POINT_DICT = {i: np.zeros(shape = (N,N)) for i in range(len(SALIENT_POINTS))}

#print(SALIENT_POINT_DICT)


print(SALIENT_POINTS)

for e in tqdm(range(n_exp)):

	s = np.array([np.random.randint(N), np.random.randint(N)])
	path = []
	
	for t in range(ep_length):
		path.append(s)

		flag = False
		for index, point in enumerate(SALIENT_POINTS):
			
			if tuple(point) == tuple(s):

				flag = True
				for ss in path:

					grid[ss[0]][ss[1]] +=1 
					SALIENT_POINT_DICT[index][ss[0]][ss[1]]+=1
				
				break

				
		if flag:
			break
		
		if not flag and t ==ep_length-1:
			
			for ss in path:
				grid[ss[0]][ss[1]] +=1 


		s = step(s)

#print(grid)

print(1*((SALIENT_POINT_DICT[2]/grid)>0))

n = np.zeros(shape = (N,N))
p = np.zeros(shape = (N,N)) 
q = SALIENT_POINT_DICT[0].copy()/grid

for point in range(len(SALIENT_POINTS)):

	n += 1*((SALIENT_POINT_DICT[point]/grid)>0)
	p += (SALIENT_POINT_DICT[point]/grid)
	q = np.minimum(q, SALIENT_POINT_DICT[point]/grid)

print(q)
U = np.power(2,n)*(1 + np.nan_to_num(p/n))
U = np.power(2,n)*(1 + q)
U = np.nan_to_num(U)
print(U)
plt.imshow(np.power(2,U-16), cmap='Blues', interpolation='nearest')
# Add colorbar 
plt.colorbar() 
plt.show()