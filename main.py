from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import qca

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
    
outputs = [0,1,1,0,1,0,0,1]

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
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalCircuit(individual):
	func = toolbox.compile(expr=individual)
	fitness = sum(func(*in_) == out for in_, out in zip(inputs, outputs)),
	height = individual.height
	if fitness[0] == 2**NO_OF_INPUTS and height != 0:
		fitness = list(fitness)
		fitness[0] += float(1/height)
		fitness = tuple(fitness)		
	return fitness
	
toolbox.register("evaluate", evalCircuit)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
#    random.seed(10)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.8, 0.1, 50, stats, halloffame=hof)
    
    print hof[0]
      
    return pop, stats, hof

if __name__ == "__main__":
    main()
