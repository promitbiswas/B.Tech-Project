from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy

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
    
outputs = [0,0,1,1,1,1,1,1]

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

def printOnlySolutions(hof):
	sols = []
	for i in xrange(len(hof)):
		func = toolbox.compile(expr=hof[i])
		fitness = sum(func(*in_) == out for in_, out in zip(inputs, outputs)),
		if fitness[0] == 2**NO_OF_INPUTS:
			print hof[i]
			hof[i].fitness

def main():
#    random.seed(10)
    pop = toolbox.population(n=999)
    spc_ind = creator.Individual(gp.PrimitiveTree.from_string("m(B, m(A, 1, i(C)), i(m(B, A, 1)))",pset))
    pop += [spc_ind]
    hof = tools.HallOfFame(100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaMuPlusLambda(pop, toolbox, mu=20, lambda_=1000, cxpb=0.6, mutpb=0.2, ngen=500, stats=stats, halloffame=hof)
    
    """printOnlySolutions(hof)
    for i in xrange(len(sols)):
		print str(sols[i])"""

    getBestSolution(hof)
    
if __name__ == "__main__":
    main()
