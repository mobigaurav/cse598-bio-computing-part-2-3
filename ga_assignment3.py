#!/usr/bin/env python
# coding: utf-8

# # Assignment 3: Extensions to Balanced Colorings and Neutral Landscapes
# 
# ### Assignment description:
# 
# In this assignment you will be asked to extend your last assignment in graph coloring to include finding balanced colorings and exploration of neutral coloring landscapes. Again, we will use the DEAP package in python to handle the genetic algorithm (GA). 
# 
# ### Assignment goals:
# 
# 1. Use a genetic algorithm to find balanced colorings for a graph
# 2. Explore the neutrality of graph colorings
# 
# ### Assignment question overview:
# 
# 1. Write a fitness fucntion for the balanced coloring problem. [Question 1 here.](#question1)
# 2. Paste your fitness function from assignment 2. (not graded) [Question 2 here.](#question2)
# 3. Write a function to calculate the 1-step neutrality a graph. [Question 3 here.](#question3)
# 

# ## Balanced Coloring Problem
# 
# In the previous assignment our goal was to find any graph coloring such that no two adjacent vertices have the same color. In the balanced coloring problem, we will keep this definition but add a second constraint. That is a coloring in which no particular color is used more than the rest. To implement this problem we will assume that the inputs and graph representation will be the same as the previous assignment. In the balanced graph coloring problem the only thing that will change is the fitness function. 
# 
# To start, we will reuse the same set up from the previous assignment. This time the ``init_graph()`` function will be supplied for you. 

# In[1]:


import random
import numpy as np
from deap import base, creator, tools, algorithms
import math

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)

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
        self.graph[s] = node  # updates the graph list to include the new node (which points to the rest of the list)

        node = AdjNode(s)
        node.next = self.graph[d]
        self.graph[d] = node
        
def init_graph(file):  
    ##open file to read from
    f = open(file, "r")
    ##store n
    n = int(f.readline())
    ##store k
    k = int(f.readline())
    ##instantiate graph obj
    graph = Graph(n,k)
    
    ##get the rest of the lines
    while True:
        str = f.readline()
        ##break if line is empty
        if not str:
            break
        ##add edge to graph
        pair = str.split()
        graph.add_edge(int(pair[0]), int(pair[1]))
            
    f.close()
    return graph


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
# Recall that a balanced coloring is a coloring where no two adjacent nodes share the same color **and** no color is used more than others. To find such colorings, we will need to adjust the fitness function accordingly. To do so, we will add an extra penalty term to the original fitness function that will reduce the fitness of unbalanced colorings. Let $G=(V,E)$ be a graph with $n$ nodes and $m$ edges. If $c(i)$ is the assigned color of node $i$, then a coloring of a graph is $C(G)=\{c(0), c(1),...,c(n)\}$. For a given edge, $i,j\in E$, let the function $\delta(i,j)=1$ if $c(i)\not = c(j)$, and $\delta(i,j)=0$ if $c(i)=c(j)$. A fitness function for the balanced coloring problem then is:
# 
# $$F(C(G)) = \frac{\sum_{i,j\in E}\delta(i,j)}{m}\prod_{j=1}^{k}\frac{|V_j|}{n}$$
# 
# where $|V_j|$ is the size of the set of all nodes of color j.
# <a id='question1'></a>
# ## Question 1: 
# 
# Write a fitness function ``eval_balance(graph, indiviudal)`` that computes the fitness of an individual given a graph and an indiviudal. The input will be a graph object and a individual object. The output will be the total fitness of an individual. Use the same indiviudal encoding scheme as in assignment 2. **Remember that if $\lceil log_2(k) \rceil \not= log_2(k)$ then some color encodings will be invalid**. 

# In[65]:


def eval_balance(graph, individual):
    '''
    Function to compute the fitness of an individual for the balanced coloring problem. 
    Inputs: 
        individual: individual object from DEAP toolbox
        graph: graph object containing nodes and edges
    Outputs:
        fitness: fitness of an individual coloring
    '''
    # your code here
    n = graph.n
    k = graph.k
    bits_per_color = math.ceil(math.log2(k))

    # Ensure the individual has the correct length
    if len(individual) != n * bits_per_color:
        return (0,)
    
    def iter_adj_nodes(graph, vertex):
        node = graph.graph[vertex]
        while node is not None:
            yield node
            node = node.next
            
    # Helper function to extract the color of a node from the individual
    def get_color(node_idx):
        start = node_idx * bits_per_color
        end = start + bits_per_color
        color = int(''.join(map(str, individual[start:end])), 2)
        # Check for invalid color and handle it
        if color >= k:
            return -1  # Invalid color code
        return color
    
    color_counts = [0] * k
    valid_edges = 0
    
    for i in range(n):
        current_color = get_color(i)
        color_counts[current_color] += 1
        for adj_node in iter_adj_nodes(graph, i):
            if current_color != get_color(adj_node.vertex):
                valid_edges += 1
    
    valid_edges /= 2  # Adjust for double counting

    color_proportions_product = 1
    for count in color_counts:
        if count != 0:  # Only consider colors that have been used
            color_proportions_product *= (count / n)
    
    m = sum(1 for _ in range(graph.n) for _ in iter_adj_nodes(graph, _)) / 2
    fitness = (valid_edges / m) * color_proportions_product
    return (fitness,)


# Your code should return (0.03125,) for input [0, 1, 0, 0, 1, 0, 0, 1]

# In[66]:


graph = init_graph("graphs/graph_1.txt")
register_ind(graph)

ind = creator.Individual([0, 1, 0, 0, 1, 0, 0, 1])

eval_balance(graph, ind)


# In[58]:


'''
This cell contains hidden tests, which are run on submission.
Test cases 1-10 correspond to graphs/graph_1.txt, test cases 11-20 correspond to graphs/graph_2.txt.
'''
 


# ### Evolving the population
# 
# Now that we have all of the necessary functions to create a graph and calculate the fitness of an indiviudal, we can create our population and allow it to evolve. Similar to assignment 2, there are 4 graphs provided. We will need to write a main function that takes a path to the graph file and finds the coloring with the highest fitness. 

# In[67]:


def main(file):
    import numpy
    
    graph = init_graph(file)
    register_ind(graph)
    
    toolbox.register("evaluate", eval_balance, graph)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, stats=stats, halloffame=hof, verbose=True)
    
    return pop, logbook, hof


# In[68]:


if __name__ == "__main__":
    pop, log, hof = main("graphs/graph_2.txt")
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


# ## Neutral Landscapes
# 
# In this section of the assignment we will explore the neutrality of graph colorings. A mutation to an individual is *neutral* if it does not affect the fitness of that individual.
# 
# First, you will need to implement 1-step neutrality. That is, starting with a single valid coloring, make copies in which one gene is changed to one of the other valid colors. Performing this process should produce $n*(k-1)$ individuals. You will then report the fraction of those individuals that have the same fitness as the original coloring.
# <a id='question2'></a>
# ## Question 2:
# 
# Below, you will copy and paste your ``eval_graph(graph, individual)`` function from assignment 2 to use as a helper function for question 3. 

# In[69]:


def eval_graph(graph, individual):
    
    n = graph.n
    k = graph.k
    bits_per_color = math.ceil(math.log2(k))
    
    def iter_adj_nodes(graph, vertex):
        node = graph.graph[vertex]
        while node is not None:
            yield node
            node = node.next
    
    def get_color(node_idx):
        start = node_idx * bits_per_color
        end = start + bits_per_color
        color = int(''.join(map(str, individual[start:end])), 2)
        # Check for invalid color and handle it
        if color >= k:
            return -1  # Invalid color code
        return color
    
    color_counts = [0] * k
    valid_edges = 0
    
    for i in range(n):
        current_color = get_color(i)
        color_counts[current_color] += 1
        for adj_node in iter_adj_nodes(graph, i):
            if current_color != get_color(adj_node.vertex):
                valid_edges += 1
    
    valid_edges /= 2  # Adjust for double counting

    color_proportions_product = 1
    for count in color_counts:
        if count != 0:  # Only consider colors that have been used
            color_proportions_product *= (count / n)
    
    m = sum(1 for _ in range(graph.n) for _ in iter_adj_nodes(graph, _)) / 2
    fitness = (valid_edges / m) * color_proportions_product
    
    return (fitness,)


# <a id='question3'></a>
# ## Question 3:
# 
# Write the function ``one_neutral(graph, individual)`` that computes the neutrality of single mutations (the fraction of mutations that result in the same fitness that the original coloring produced). The inputs will be a graph object and an individual object and the output will be the neutrality.

# In[70]:


def one_neutral(graph, individual):
    original_fitness = eval_graph(graph, individual)[0]
    counter = 0
    n = graph.n
    k = graph.k
    bits_per_color = math.ceil(math.log2(k))
    
    def get_color(node_idx):
        start = node_idx * bits_per_color
        end = start + bits_per_color
        return int(''.join(map(str, individual[start:end])), 2)
    
    def set_color(individual, node_idx, color):
        start = node_idx * bits_per_color
        end = start + bits_per_color
        binary_color = format(color, '0' + str(bits_per_color) + 'b')
        individual[start:end] = list(map(int, binary_color))
    
    total_mutations = 0
    for i in range(n):
        current_color = get_color(i)
        for new_color in range(k):
            if new_color != current_color:  # Exclude the original color
                mutated_individual = individual.copy()
                set_color(mutated_individual, i, new_color)
                mutated_fitness = eval_graph(graph, mutated_individual)[0]
                if mutated_fitness == original_fitness:
                    counter += 1
                total_mutations += 1
    return counter / total_mutations


# Your code should return 0.25 for input \[0, 1, 0, 0, 1, 0, 0, 1\]; check that it does:

# In[71]:


graph = init_graph("graphs/graph_1.txt")
register_ind(graph)

ind = creator.Individual([0, 1, 0, 0, 1, 0, 0, 1])

one_neutral(graph, ind)


# In[72]:


'''
This cell contains hidden tests, which are run on submission.
There are five test cases each for graph_1 to graph_4.
'''


# In[ ]:





# In[ ]:





# In[ ]:




