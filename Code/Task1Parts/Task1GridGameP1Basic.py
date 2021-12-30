#make game grid
import numpy as np
import pandas as pd

from random import randint
#set dimensions
width = 4
height = 4
#create grid list
gridList = []

#fill list with random int and make 2d
for column in range(width):
    gridList.append([])
    for row in range(height):
        gridList[column].append(randint(0,9))
        
# Convert list to numpy array
gridNpA = np.array(gridList)
print(gridNpA)



#Basic Algorithm class
#aim of game is to get to bottom right corner with better than random movement
class BasicAlg:
#initialise constructor
    def __init__(self, graph):
        self.graph = graph

#initialise the variables
    def alg(self):
        totTime = 0
        org = [0,0]
        curPos = org
        x = curPos[1]
        y = curPos[0]
    
#First if not at edge of grid, move right while adding the time to the total
        while(y != (width-1)):
            y += 1
            totTime += self.graph[x,y];
        
#then, if not at edge of grid, move down while adding the time to the total
        while(x != (height-1)):
            x += 1
            totTime += self.graph[x,y];
        
        return totTime
    
#run the basic algorithm on the grid
basic = BasicAlg(gridNpA)
print(basic.alg())