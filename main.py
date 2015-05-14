from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import operator
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
	
NO_OF_INPUTS = 3
inputs = [[0] * NO_OF_INPUTS for i in xrange(2 ** NO_OF_INPUTS)]
outputs = [None] * (2 ** NO_OF_INPUTS)

for i in xrange(2 ** NO_OF_INPUTS):
    value = i
    divisor = 2 ** NO_OF_INPUTS
    # Fill the input bits
    for j in xrange(NO_OF_INPUTS):
        divisor /= 2
        if value >= divisor:
            inputs[i][j] = 1
            value -= divisor
    
outputs = [1,1,0,0,0,0,1,1]

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
"""pset.renameArguments(IN3='D')"""

creator.create("Fitness", base.Fitness, weights=(10.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=0, max_=15)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalCircuit(individual):
	func = toolbox.compile(expr=individual)
	fitness = sum(func(*in_) == out for in_, out in zip(inputs, outputs))
	maj = str(individual).count('m')
	inv = str(individual).count('i')
	height = individual.height
	"""cell_count = 5*maj + 13*inv
	if fitness == 2**NO_OF_INPUTS and cell_count != 0:
		fitness += float(1/cell_count)"""
	return fitness,height
	
toolbox.register("evaluate", evalCircuit)
toolbox.register("select", tools.selRoulette)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def getBestSolution(hof):
	sol = "Solution Not Found"
	for i in xrange(len(hof)):
		func = toolbox.compile(expr=hof[i])
		fitness = sum(func(*in_) == out for in_, out in zip(inputs, outputs)),
		if fitness[0] == 2**NO_OF_INPUTS:
			sol = hof[i]
			minGates = str(hof[i]).count('m')
			minLength = len(hof[i])
			break
	for j in xrange(i+1,len(hof)):
		func = toolbox.compile(expr=hof[j])
		fitness = sum(func(*in_) == out for in_, out in zip(inputs, outputs)),
		gateCount = str(hof[j]).count('m')
		if fitness[0] == 2**NO_OF_INPUTS and gateCount < minGates:
			sol = hof[j]
			minGates = gateCount
			minLength = len(hof[j])
		elif fitness[0] == 2**NO_OF_INPUTS and gateCount == minGates and len(hof[j]) < minLength:
			sol = hof[j]
			minGates = gateCount
			minLength = len(hof[j])
	print str(sol)
	print sol.height
	print sol.fitness
	
def printOnlySolutions(hof):
	sols = []
	for i in xrange(len(hof)):
		func = toolbox.compile(expr=hof[i])
		fitness = sum(func(*in_) == out for in_, out in zip(inputs, outputs)),
		if fitness[0] == 2**NO_OF_INPUTS:
			print hof[i]

def main():
#    random.seed(10)
    pop = toolbox.population(n=2000)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaMuPlusLambda(pop, toolbox, mu=1000, lambda_=2000, cxpb=0.65, mutpb=0.35, ngen=1000, stats=stats, halloffame=hof)
    
    """printOnlySolutions(hof)"""
    getBestSolution(hof)
    
if __name__ == "__main__":
    main()
