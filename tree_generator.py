from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv

NO_OF_INPUTS = 3
inputs = [[0] * NO_OF_INPUTS for i in xrange(2 ** NO_OF_INPUTS)]
outputs = [None] * (2 ** NO_OF_INPUTS)

def m(a,b,c):
	return (a&b|b&c|a&c)

def i(a):
	return (~a) 

pset = gp.PrimitiveSet("MAIN", NO_OF_INPUTS, "IN")
pset.addPrimitive(m, 3)
pset.addPrimitive(i, 1)
pset.addTerminal(1)
pset.addTerminal(0)
pset.renameArguments(IN0='A')
pset.renameArguments(IN1='B')
pset.renameArguments(IN2='C')

creator.create("FitnessMax", base.Fitness, weights=(10.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=0, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalCircuit(individual):
	func = toolbox.compile(expr=individual)
	fitness = sum(func(*in_) == out for in_, out in zip(inputs, outputs)),
	height = individual.height
	length = len(individual)
	if fitness[0] == 2**NO_OF_INPUTS and height != 0:
		fitness = list(fitness)
		fitness[0] += float(1/height)
		fitness = tuple(fitness)		
	return fitness
	
toolbox.register("evaluate", evalCircuit)
toolbox.register("select", tools.selDoubleTournament, fitness_size=300, parsimony_size=2, fitness_first=True)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

ind = creator.Individual(gp.PrimitiveTree.from_string("m(1,m(B,A,0),i(m((A,B,1)))",pset))
nodes,edges,labels = gp.graph(ind)
g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")
for j in nodes:
	n = g.get_node(j)
	n.attr["label"] = labels[j]
name = "ind.fig"
g.draw(name)
