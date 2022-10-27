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

    def neighbors(self, id):
        # check if the edge is in the edge dictionary
        if id in self.edges:
            return self.edges[id]
        else:
            print("The node ", id , " is not in the graph")
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

    #print ("EXECUTING RECONSTRUCT")

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
    # return path

def breadth_first_search(graph, start, goal):
    """
    Given a graph, a start node and a goal node
    Utilize breadth first search algorithm by finding the path from
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

    #print ("EXECUTING BFS")

    came_from = {}
    came_from[start] = None
    # initialize a queue and place the start node in the queue
    q = Queue()
    q.put(start)
    visited = set()
    #print("start is " + start + " goal is " + goal )
    ### START CODE HERE ### (≈ 10 line of code)
    ### while the queue isn't empty connected nodes from the node popped from the queue
    while q.qsize() != 0:

        currNode = q.get()
        #print("currNode is " + currNode)
        if currNode not in visited:
            ### add node popped to the visited list
            visited.add(currNode)
            neighbours = graph.neighbors(currNode)
            for neighbour in neighbours:
                #print(neighbour)
                ### queue neighbour
                q.put(neighbour)
                if neighbour not in visited:
                    came_from[neighbour] =  currNode
                ### if the edge is the goal we've reached the goal and we return came_from
                if neighbour == goal:
                    return came_from
        #print("\n")

    ### END CODE HERE ###
    return came_from

def depth_first_search(graph, start, goal):
    """
    Given a graph, a start node and a goal node
    Utilize depth first search algorithm by finding the path from
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
    #print ("EXECUTING DFS")

    came_from = {}
    came_from[start] = None
    # initialize a queue and place the start node in the queue
    s = LifoQueue()
    s.put(start)
    visited = set()
    #print("start is " + start + " goal is " + goal )
    ### START CODE HERE ### (≈ 10 line of code)
    ### while the queue isn't empty connected nodes from the node popped from the queue
    while s.qsize() != 0:

        currNode = s.get()
        #print("currNode is " + currNode)
        if currNode not in visited:
            ### add node popped to the visited list
            visited.add(currNode)
            neighbours = graph.neighbors(currNode)
            for neighbour in neighbours:
                #print(neighbour)
                ### queue neighbour
                s.put(neighbour)
                if neighbour not in visited:
                    came_from[neighbour] =  currNode
                ### if the edge is the goal we've reached the goal and we return came_from
                if neighbour == goal:
                    return came_from
        #print("\n")

    ### END CODE HERE ###
    return came_from


# The main function will first create the graph, then use depth first search
# and breadth first search which will return the came_from dictionary
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
    large_graph = Graph()
    large_graph.edges = {
        'S': ['A','C'],
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
    print("Large graph")
    start = 'S'
    goal = 'E'
    came_fromDFS = depth_first_search(large_graph, start, goal)
    print("came from DFS" , came_fromDFS)
    pathDFS = reconstruct_path(came_fromDFS, start, goal)
    print("path from DFS", pathDFS)
    came_fromBFS = breadth_first_search(large_graph, start, goal)
    print("came from BFS", came_fromBFS)
    pathBFS = reconstruct_path(came_fromBFS, start, goal)
    print("path from BFS", pathBFS)

    print("Small graph")
    start = 'A'
    goal = 'E'
    came_fromDFS = depth_first_search(small_graph, start, goal)
    print("came from DFS" , came_fromDFS)
    pathDFS = reconstruct_path(came_fromDFS, start, goal)
    print("path from DFS", pathDFS)
    came_fromBFS = breadth_first_search(small_graph, start, goal)
    print("came from BFS", came_fromBFS)
    pathBFS = reconstruct_path(came_fromBFS, start, goal)
    print("path from BFS", pathBFS)
