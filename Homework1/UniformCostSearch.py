# -*- coding: utf-8 -*-
from queue import LifoQueue
from queue import Queue
from queue import PriorityQueue

class Graph:
    """
    Defines a graph with edges, each edge is treated as dictionary
    look up. function neighbors pass in an id and returns a list of
    neighboring node

    """
    def __init__(self):
        self.edges = {}
        self.edgeWeights = {}
        self.locations = {}

    def neighbors(self, id):
        if id in self.edges:
            return self.edges[id]
        else:
            print("The node ", id , " is not in the graph")
            return False

    def get_node_location(self, id):
        return self.nodeLocation[id]

    def get_cost(self,from_node, to_node):
        #print("get_cost for ", from_node, to_node)
        nodeList = self.edges[from_node]
        #print(nodeList)
        try:
            edgeList = self.edgeWeights[from_node]
            return edgeList[nodeList.index(to_node)]
        except ValueError:
            print("From node ", from_node, " to ", to_node, " does not exist a direct connection")
            return False

def reconstruct_path(came_from, start, goal):
    """
    Given a dictionary of came_from where its key is the node
    character and its value is the parent node, the start node
    and the goal node, compute the path from start to the end

    Arguments:
    came_from -- a dictionary indicating for each node as the key and
                 value is its parent node
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    path. -- A list storing the path from start to goal. Please check
             the order of the path should from the start node to the
             goal node
    """
    path = []
    visited = set()
    ### START CODE HERE ### (≈ 6 line of code)
    currNode = goal
    path.append(goal)
    while 1:
        currNode = came_from.get(currNode)
        if currNode in visited:
            return "You're in a loop! Cannot find start node. Exiting"
        else:
            path.append(currNode)
            visited.add(currNode)
        if currNode == start:
            break
    ### END CODE HERE ###
    return list(reversed(path))

    ### END CODE HERE ###
    # return path


def uniform_cost_search(graph, start, goal):
    """
    Given a graph, a start node and a goal node
    Utilize uniform cost search algorithm by finding the path from
    start node to the goal node
    Use early stoping in your code
    This function returns back a dictionary storing the information of each node
    and its corresponding parent node
    Arguments:
    graph -- A dictionary storing the edge information from one node to a list
             of other nodes
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    came_from -- a dictionary indicating for each node as the key and
                value is its parent node
    """
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    visited = set()
    q = PriorityQueue()
    q.put((0, start))

    while q.qsize() != 0:
        currNode = q.get()
        #print("currNode is " + currNode[1] + " and the cost is " + str(currNode[0]))
        currNode = currNode[1]
        if currNode not in visited:
            
            visited.add(currNode)
            if currNode == goal:
                return came_from, cost_so_far

            neighbours = graph.neighbors(currNode)
            for neighbour in neighbours:
                #print(neighbour)
                if neighbour not in came_from.keys():
                    q.put((cost_so_far[currNode] +  graph.get_cost(currNode, neighbour), neighbour))
                    came_from[neighbour] =  currNode
                    cost_so_far[neighbour] = cost_so_far[currNode] +  graph.get_cost(currNode, neighbour)
                elif cost_so_far[currNode] +  graph.get_cost(currNode, neighbour) < cost_so_far[neighbour]:
                    q.put((cost_so_far[currNode] +  graph.get_cost(currNode, neighbour), neighbour))
                    came_from[neighbour] =  currNode
                    cost_so_far[neighbour] = cost_so_far[currNode] +  graph.get_cost(currNode, neighbour)
                else:
                    continue
        #print("\n")

    ### END CODE HERE ###
    return came_from, cost_so_far

# The main function will first create the graph, then use uniform cost search
# which will return the came_from dictionary
# then use the reconstruct path function to rebuild the path.
if __name__=="__main__":
    small_graph = Graph()
    small_graph.edges = {
        'A': ['B','D'],
        'B': ['A', 'C', 'D'],
        'C': ['A'],
        'D': ['E', 'A'],
        'E': ['B']
    }
    small_graph.edgeWeights={
        'A': [2,4],
        'B': [2, 3, 4],
        'C': [2],
        'D': [3, 4],
        'E': [5]
    }

    large_graph = Graph()
    large_graph.edges = {
        'S': ['A','B','C'],
        'A': ['S','B','D'],
        'B': ['S', 'A', 'D','H'],
        'C': ['S','L'],
        'D': ['A', 'B','F'],
        'E': ['G','K'],
        'F': ['H','D'],
        'G': ['H','E'],
        'H': ['B','F','G'],
        'I': ['L','J','K'],
        'J': ['L','I','K'],
        'K': ['I','J','E'],
        'L': ['C','I','J']
    }
    large_graph.edgeWeights = {
        'S': [7, 2, 3],
        'A': [7, 3, 4],
        'B': [2, 3, 4, 1],
        'C': [3, 2],
        'D': [4, 4, 5],
        'E': [2, 5],
        'F': [3, 5],
        'G': [2, 90],
        'H': [1, 3, 2],
        'I': [4, 6, 4],
        'J': [4, 6, 4],
        'K': [4, 4, 5],
        'L': [2, 4, 4]
    }

    print("Small graph")
    start = 'A'
    goal = 'E'
    came_from_UCS, cost_so_far = uniform_cost_search(small_graph, start, goal)
    print("came from UCS " , came_from_UCS)
    print("cost form UCS ", cost_so_far)
    pathUCS = reconstruct_path(came_from_UCS, start, goal)
    print("path from UCS ", pathUCS)

    print("Large graph")
    start = 'S'
    goal = 'E'
    came_from_UCS, cost_so_far = uniform_cost_search(large_graph, start, goal)
    print("came from UCS " , came_from_UCS)
    print("cost form UCS ", cost_so_far)
    pathUCS = reconstruct_path(came_from_UCS, start, goal)
    print("path from UCS ", pathUCS)
