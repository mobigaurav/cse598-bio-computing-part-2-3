#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Introduction to Graph Coloring 
# 
# ### Assignment description:
# 
# In this assignment you will learn more about genetic algorithms and use your knowledge to solve the graph coloring problem. You will be asked to implement a solution to this problem using the DEAP package from assignment 1. 
# 
# ### Assignment goals:
# 
# 1. Understand the graph coloring problems
# 2. Learn how to represent graphs in DEAP
# 3. Write the fitness function for the graph coloring problem
# 4. Implement a genetic algorithm to find graph colorings
# 
# ### Assignment question overview:
# 
# 1. Write a function to read in a graph file as a Graph object. [Question 1 here.](#question1)
# 2. Write a function to compute the fitness of an individual. [Question 2 here.](#question2)
# 3. Find a new parameterization of the model for a large graph. [Question 3 here.](#question3)
# 

# ## Graph Coloring Problem
# 
# We will use genetic algorithms (GAs) to solve the graph coloring problem. The problem takes as input a graph $G=(V,E)$ and an integer $k$ which designates the numbers of different colors (labels) to be used. A sucessful coloring assigns each vertex $v\in V$ a color $\in \{1,2,3,...,k\}$ such that all adjacent vertices have different colors (i.e. for all vertices $v\in V$ if there exists an edge $e \in E$ between $v_1$ and $v_2$ then $v_1$ and $v_2$ must have different colors). The general problem of graph coloring is computationally intractable for $k>2$, so we will use the GA to search heuristically for solutions. 

# <a id='representation'></a>
# ## Representation
# 
# The input for each graph will be a file in the `graphs/` directory. The first line of each file will indicate how many nodes there are in the graph, $n$, and the second line will indicate how many colors to use, $k$, and the remainder of the file will be a list of edges represented as ordered pairs (the two connected vertices seperated by a space) with ones edge per line. 
# 
# The following figure shows the contents of `graphs/graph_1.txt` on the left, a drawing of the uncolored graph in the middle, and a drawing of the colored graph on the right. Note that colorings for the graphs are not necessarily unique.
# 
# <img src="images/graph1.png" alt="Drawing" style="width:800px;"/>
# 
# You can assume that the nodes in any input graph are each labeled with a unique (for the node) positive integer starting at zero. Thus, in the example graph above there are four vertices, connected by three edges, and the task is to find a 3-coloring of the graph. 
# 
# Your GA will maintain a population of individuals, where each individual specifies a candidate coloring for the graph. The most natural representation of a candidate graoh coloring along the chromosome is one where the first position corresponds to Node 1, the second to Node 2, etc. Then the value (allele) at each position will correspond to the color. 
# 
# ### Imports 
# 
# We will first import the required modules for this notebook. 
# 
# - `random` gives us a way to generate random bits;
# - `numpy` supports mathematical operations in multidimensional arrays;
# - `base` gives us access to the Toolbox and base Fitness;
# - `creator` allows us to create our types;
# - `tools` grants us access to the operators bank;
# - `algorithms` enables us some ready generic evolutionary loops;
# - `math` allows us to use basic mathematical functions.

# In[10]:


import random
import numpy as np
from deap import base, creator, tools, algorithms
import math


# ### Setting up
# 
# We will set up our GA by first creating the same types used in assignment 1. 

# In[11]:


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# ### File I/O and Graph Data Structure
# 
# Before creating individuals, we first must read in an arbitrary graph from the `graphs/` directory into a data structure. Being that the data is already represented as an adjacency matrix, we will use an adjacency list representation to easily access the graph.  An adjacency list represents a graph as an array of linked lists. Each node in the graph will have a linked list of all its adjacent nodes. The structure of this list will allow for an efficient way to check if two nodes share an edge. The figure below shows the example graph given before represented as an adjacency list.
# 
# <img src="images/adj_list.png" alt="Drawing" style="width:800px;"/>
# 
# To support this representation we will use two classes. First, the `AdjNode` class will be used to represent the nodes of the graph. This class has three attributes, the value of the vertex, the first adjacent node in the list. The second class is `Graph` and will be used to store all of the nodes and their adjacency lists. The `Graph` class has four attributes, the number of nodes $n$, the number of colors $k$, the array `graph` that will store the adjacency matrix of nodes, and `color_assignment`, a list you will need to use to assign a color to each node. Additionally, there is a helper function `add_edge(s,d)`, that when supplied with two nodes (`s` and `d`) will add an edge to the graph.
# 
# **To use this structure to check if, say, an edge exists between node 2 and node 3, you can access the Graph.graph\[2\] (a linked list of the neighbors of node 2) and traverse it, checking if any of the linked nodes are node.vertex == 3. The symmetric case also works (accessing Graph.graph at 3 and checking for 2 in the linked list of neighbors). To assign a color to a node (e.g., coloring node 4 with color 7) you can use Graph.color_assignment\[4\] = 7**
# 

# In[12]:


class AdjNode:
    def __init__(self, value):
        self.vertex = value
        self.next = None


class Graph:
    def __init__(self, num, colors):
        self.n = num
        self.k = colors
        self.graph = [None] * self.n
        self.color_assignment = [None] * self.n
        
    def get_node(self, s):
        return self.graph[s]
        
    # Add edges
    def add_edge(self, s, d):
        node = AdjNode(d)  # creates a new node for the destination half of the edge
        node.next = self.graph[s]  # connects the newly-created node to the existing linked list (at the front)
        self.graph[s] = node  # updates the list to start with the new node (points to the rest of the list)

        node = AdjNode(s)
        node.next = self.graph[d]
        self.graph[d] = node


# <a id='question1'></a>
# ## Question 1: 
# 
# Write a function `init_graph(file)` that creates a graph from a text file. Recall that the first line of the text file will be an integer value, $n$, that represents the number of nodes in the graph $G=(V,E)$ and the second line will be an integer value, $k$, that represents the number of colors to be used. The following lines will be ordered pairs seperated by a space delimeter, for each ordered pair in the text file there will be a corresponding edge between the given nodes. Refer to the [Representation](#representation) section for more details. 
# 
# The input of `init_graph(file)` will be a path to a graph file in the `graphs/` directory (ex. `graphs/graph_1.txt`) and the output will be a `Graph` object. 

# In[13]:


def init_graph(file):
    '''
    Function to read in graph file and create a graph object. 
    Inputs: 
        file: path to a graph file in /graphs
    Outputs:
        graph: Graph object containing edges given in file
    '''
    
    # your code here
    with open(file, 'r') as f:
        #parse number of nodes and colors
        n = int(f.readline().strip())
        k = int(f.readline().strip())
        print(n,k)
        #create graph object with n nodes and k colors
        graph = Graph(n,k)
        #parse edge and add them to graph
        for line in f:
            s,d = map(int, line.split())
            print(s,d)
            graph.add_edge(s,d)
    
    return graph


# In[ ]:


'''
This cell contains hidden tests, which are run on submission.
Hidden test cases test on graphs in the graphs/ directory. 
'''


# ### Creating Individuals
# 
# As mentioned prior, the most natural representation of a candidate graph coloring along the chromosome is one where the first position corresponds to Node 1, the second to Node 2, etc. Then the value (allele) at each position will correspond to the color. This bit string encoding will require that you allocate $\lceil log_2k \rceil$ bits for each gene. This scheme has the drawback that for some values of $k$ not all bit values will correspond to a legal color. Those individuals with an invalid coloring will have a fitness of 0. 
# 
# The figure below shows this encoding scheme on our example graph. Here, $k=3$ and $n=4$. This means that each individual will be lenghth $l=4\lceil log_2k \rceil$. In the example, the substring 00 encodes red, 01 encodes blue, and 10 encodes for green. Since 3 is not a power of 2, the substring 11 is not used.
# 
# <img src="images/encoding.png" alt="Drawing" style="width:800px;"/>
# 
# We will need to register the boolean attribure as well as the individual and population with the DEAP toolbox. We first instantiate the toolbox and then register the boolean attributes since they are both independent of the given graph. Next, we create the function `register_ind(graph)` which takes a graph and registers the correct size of individual to the toolbox. This function will also register the population given the indiviudal. This function will be used later in the main method, but will be helpful now in testing the fitness function. 

# In[14]:


toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)

def register_ind(graph):
    ##get value for n and k from graph
    n = graph.n
    k = graph.k
    
    ##calculate the size of each individual
    ind_size = math.ceil(math.log2(k))*n
    
    ##register individual and population with toolbox
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ### Fitness Function
# 
# For each individual, we want to know how close its encoded coloring is to a legal coloring of the graph (recall, a legal coloring is one where no nodes connected by an edge have the same color). Let $G=(V,E)$ be a graph with $n$ nodes and $m$ edges. If $c(i)$ is the assigned color of node $i$, then a coloring of a graph is $C(i)=\{c(0),c(1),...c(n)\}$. For a given edge $i,j\in E$, let the function $\delta(i,j) = 1$ if $c(i)\not=c(j)$, and $\delta(i,j)=0$ if $c(i)=c(j)$. A natural fitness function then is:
# 
# $$F(C(G)) = \frac{\sum_{i,j\in E} \delta(i,j)}{m}$$
# 
# That is the fraction of edges where the endpoints do not share the same color, divided by the total number of edges.
# <a id='question2'></a>
# ## Question 2:
# 
# Write the fitness function `eval_graph(graph, individual)` that computes the fitness of an individual given a graph and individual. The input will be a graph object and an individual object. The output will be the total fitness of the indiviudal. **Remember that if $\lceil log_2(k) \rceil \not= log_2(k)$ then some color encodings will be invalid**.

# In[30]:


def decode(individual, k):
    """
    Decode a binary individual into its color representation.

    Args:
    - individual: A binary list representing the individual.
    - k: Number of colors.
    
    Returns:
    - A list of colors for each node.
    """
    # Calculate the length of binary representation for each color
    length = math.ceil(math.log2(k))
    
    # Split the individual into chunks of the calculated length
    chunks = [individual[i:i+length] for i in range(0, len(individual), length)]
    
    # Convert each chunk into its decimal color representation
    colors = [int(''.join(map(str, chunk)), 2) for chunk in chunks]
    
    return colors

def eval_graph(graph, individual):
    '''
    Function to compute the fitness of an individual for the graph coloring problem. 
    Inputs: 
        individual: individual object from DEAP toolbox
        graph: graph object containing nodes and edges
    Outputs:
        fitness: fitness of an individual coloring
    '''
    k = graph.k
    colors = decode(individual, k)
    print("colors", colors)
    # If the decoded colors contain invalid colors, return a low fitness value
    if any(color >= k for color in colors):
        return (0,)

    m = 0  # Count of total edges
    correct_edges = 0  # Count of edges where nodes do not have the same color

    for i in range(graph.n):
        node = graph.get_node(i)
        while node:
            m += 1
            if colors[i] != colors[node.vertex]:
                correct_edges += 1
            node = node.next

    # Compute the fitness value
    fitness = correct_edges / m
    
    return (fitness,)


# Your function should return (0.6666666666666666,) for inputs below:

# In[31]:


graph = init_graph("graphs/graph_1.txt")
register_ind(graph)

ind = creator.Individual([1, 0, 0, 0, 1, 0, 0, 1])
eval_graph(graph, ind)


# In[ ]:


'''
This cell contains hidden test cases, which are run on submission.
Tests 1-10 are on graphs/graph_1.txt; tests 11-20 are on graphs/graph_2.txt.
'''


# ### Update Toolbox and Evolve Population
# 
# Now that we have all of the necessary functions to create a graph and calculate the fitness of an individual, we can create our population and allow it to evolve. To do this, we will need to write a main function that takes a path to a graph file and finds the coloring with the highest fitness. Note that the range of the fitness function is $[0,1]$. 
# <a id='question3'></a>
# ## Question 3: 
# 
# The parameterization that we used in the previous assignment will not be sufficient to guarantee (with a high probability) convergence to a fitness value of 1 for larger graphs such as ``graph_4.txt``. Before running the main function, we will need to update the parameterization of the model. Recall from assignment 1 that ``cxpb`` is the probability with which two individuals are crossed, ``mutpb`` is the probability for mutating an individual, ``indpb`` is the the independent probability of each attribute to be mutated, ``tournsize`` is the size of each tournament, ``n`` is the population size, and  ``ngen`` is the number of generations. Below, set the values of these parameters such that a run of the main function will return a individual with fitness 1 for ``graph_4.txt``. 

# In[32]:


def main(file):
    import numpy
    
    '''
    Set the following parameters: indpb, tournsize, cxpb, mutpb, ngen, and n. 
    Below is the orginal parameterization of the model:
        indpb = 0.10
        tournsize = 3
        cxpb = 0.5
        mutpb = 0.2
        ngen = 30
        n=50
    '''
    
    # your code here
    indpb = 0.02      # Decrease the bit mutation probability due to larger individual size
    tournsize = 5     # Keep a higher tournament size for better selection pressure
    cxpb = 0.8        # Increase crossover probability to promote genetic diversity
    mutpb = 0.3       # Slight increase in mutation probability to ensure a mix of individuals
    ngen = 200        # Significantly increase the number of generations to allow more iterations
    n = 300           # Increase the population size to ensure diverse initial population

    
    graph = init_graph(file)
    register_ind(graph)
    
    toolbox.register("evaluate", eval_graph, graph)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    
    pop = toolbox.population(n=n)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof, verbose=True)
    
    
    return pop, logbook, hof


# In[33]:


if __name__ == "__main__":
    pop, log, hof = main("graphs/graph_4.txt")
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    
    
    import matplotlib.pyplot as plt
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:




