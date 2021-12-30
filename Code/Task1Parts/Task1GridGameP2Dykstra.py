#Import necessary libraries
import numpy as np
import pandas as pd
import sys as sy

from random import randint

#set dimensions, origin and end positions
width = 4
height = 4
org = [0,0]
end = [height, width]
#create grid list
gridList = []

#fill list with random int and make it 2d
for column in range(width):
    gridList.append([])
    for row in range(height):
        gridList[column].append(randint(0,9))

# Convert 2d list to numpy array
gridNpA = np.array(gridList)
print(gridNpA)

#Basic Algorithm class
#aim of game is to get to bottom right corner with better than random movement
#first move right all the way. then move down all the way. this cuts out random backward moves
class BasicAlg():
#Initialise constructor
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
#Time taken per step added to total
            totTime += self.graph[x,y];
        
#then, if not at edge of grid, move down while adding the time to the total
        while(x != (height-1)):
            x += 1
#Time taken per step added to total
            totTime += self.graph[x,y];
  
        return totTime

#run the basic algorithm on the grid
basic = BasicAlg(gridNpA)
print(basic.alg())

#convert grid to node
class convertGrid:
#initialise constructor    
    def __init__(self, gridNpA, origin, end):
        self.gridNpA = gridNpA
        self.org = origin
        self.end = end
#get name of node    
    def get_name(self, location):
        node_name = (location[0] * 100) + location[1]
        return str(node_name)
#get adjacent nodes    
    def get_adjacent(self, location):
        adjacent = []
        if (location[0] + 1) < (height):
            adjacent.append([location[0] + 1, location[1]])
                              
        if (location[0] - 1) >= 0:
            adjacent.append([location[0]-1, location[1]])
            
        if (location[1] + 1) < (width):
            adjacent.append([location[0], location[1] + 1])
            
        if (location[1] - 1) >= 0:
            adjacent.append([location[0], location[1] - 1])
        
        return adjacent
    
#get edges between nodes or spaces in the grid    
    def get_edges(self, location):
        adjacent = self.get_adjacent(location)
        edges = []        
        for i in adjacent:
            edges.append(self.gridNpA[i[0], i[1]])
            
        return edges      

#create nodes using their names and adjacent edges       
    def create_node(self, location):        
        adjacent = self.get_adjacent(location)
        edges = self.get_edges(location)
        adjacent_names = []
        node_name = self.get_name(location)
#possible error below with nodename sorted              
        for i in adjacent:
            adjacent_names.append(self.get_name(i))
            
        adjacent_edges = dict(zip(adjacent_names, edges))        
        return node_name, adjacent_edges

#convert grid to nodes form    
    def grid_toNode(self):
        nodes = dict()
        for i in range(self.end[0]):
            for j in range(self.end[1]):
                node = self.create_node([i,j])
                nodes[node[0]] = node[1]
        
        return nodes
		
#Dijkstra algorithm for shortest path function
#Aim of game is to get to bottom right corner with shortest path using Dykstras Algorithm
def dijkstraAlg(graph,current,last,V = [],D = dict(),P = dict()):
#visited nodes list, dictionary of distances, dictionary of paths
#if not yet visited, set distance to zero
    if not V: D[current]=0
#if we've reached the end node, return the path
    if current==last:
        path=[]
        while last != None:
            path.append(last)
            last=P.get(last,None)
        return D[current], path[::-1]
#proces adjacent nodes and keep track of visited node and paths
    for adjacent in graph[current]:
        if adjacent not in V:
            adjacentDist = D.get(adjacent,sy.maxsize)
            tempDist = D[current] + graph[current][adjacent]
            if tempDist < adjacentDist:
                D[adjacent] = tempDist
                P[adjacent] = current
#add current node as visited 
    V.append(current)
# find the closest unvisited node to current node 
    unvisitedNodes = dict((u, D.get(u,sy.maxsize)) for u in graph if u not in V)
    closestNode = min(unvisitedNodes, key=unvisitedNodes.get)
#recurse to make closest node current node 
    return dijkstraAlg(graph,closestNode,last,V,D,P)
	
#current and end string
C = (org[0] * 100) + org[1]
E = ((end[0] * 100) + end[1]) - 101

#save converted grid to nodes from in graph
CG = convertGrid(gridNpA, org, end)
graph = CG.grid_toNode()

#find dijsktras shortest path output    
print (dijkstraAlg(graph,str(C),str(E)))