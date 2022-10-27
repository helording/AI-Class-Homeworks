# -*- coding: utf-8 -*-
from queue import LifoQueue
from queue import Queue
from queue import PriorityQueue
from math import sqrt

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

    # this function get the g(n) the cost of going from from_node to
    # the to_node
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
    ### START CODE HERE ### (≈ 6 line of code)
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

def heuristic(graph, current_node, goal_node):
    """
    Given a graph, a start node and a next nodee
    returns the heuristic value for going from current node to goal node
    Arguments:
    graph -- A dictionary storing the edge information from one node to a list
             of other nodes
    current_node -- A character indicating the current node
    goal_node --  A character indicating the goal node

    Return:
    heuristic_value of going from current node to goal node
    """
    ### START CODE HERE ### (≈ 15 line of code)
    current_node_location = graph.locations[current_node]
    x1 = current_node_location[0]
    y1 = current_node_location[1]

    goal_node_location = graph.locations[goal_node]
    x2 = goal_node_location[0]
    y2 = goal_node_location[1]

    ### END CODE HERE ###
    return sqrt( ((x1-x2) ** 2) + ((y1-y2) ** 2) )

def A_star_search(graph, start, goal):
    """
    Given a graph, a start node and a goal node
    Utilize A* search algorithm by finding the path from
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

    ### START CODE HERE ### (≈ 15 line of code)
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
                # if it is the first instance of visiting the node add to dictionaries
                if neighbour not in came_from.keys():
                    q.put((cost_so_far[currNode] +  graph.get_cost(currNode, neighbour) + heuristic(graph, neighbour, goal), neighbour))
                    came_from[neighbour] =  currNode
                    cost_so_far[neighbour] = cost_so_far[currNode] +  graph.get_cost(currNode, neighbour)
                # else compare it to the cost of the previous path
                elif cost_so_far[currNode] +  graph.get_cost(currNode, neighbour) < cost_so_far[neighbour]:
                    q.put((cost_so_far[currNode] +  graph.get_cost(currNode, neighbour) + heuristic(graph, neighbour, goal), neighbour))
                    came_from[neighbour] =  currNode
                    cost_so_far[neighbour] = cost_so_far[currNode] +  graph.get_cost(currNode, neighbour)
                else:
                    continue
        #print("\n")

    ### END CODE HERE ###
    return came_from, cost_so_far


# The main function will first create the graph, then use A* search
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
    small_graph.locations={
        'A': [4,4],
        'B': [2,4],
        'C': [0,0],
        'D': [6,2],
        'E': [8,0]
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
        'G': [2, 2],
        'H': [1, 3, 2],
        'I': [4, 6, 4],
        'J': [4, 6, 4],
        'K': [4, 4, 5],
        'L': [2, 4, 4]
    }

    large_graph.locations = {
        'S': [0, 0],
        'A': [-2,-2],
        'B': [1,-2],
        'C': [6,0],
        'D': [0,-4],
        'E': [6,-8],
        'F': [1,-7],
        'G': [3,-7],
        'H': [2,-5],
        'I': [4,-4],
        'J': [8,-4],
        'K': [6,-7],
        'L': [7,-3]
    }
    print("Small Graph")
    start = 'A'
    goal = 'E'
    came_from_Astar, cost_so_far = A_star_search(small_graph, start, goal)
    print("came from Astar " , came_from_Astar)
    print("cost form Astar ", cost_so_far)
    pathAstar = reconstruct_path(came_from_Astar, start, goal)
    print("path from Astar ", pathAstar)


    print("Large Graph")
    start = 'S'
    goal = 'E'
    came_from_Astar, cost_so_far = A_star_search(large_graph, start, goal)
    print("came from Astar " , came_from_Astar)
    print("cost form Astar ", cost_so_far)
    pathAstar = reconstruct_path(came_from_Astar, start, goal)
    print("path from Astar ", pathAstar)
