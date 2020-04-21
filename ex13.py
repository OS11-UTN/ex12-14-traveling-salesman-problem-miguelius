import operator
import os
import time

import numpy as np

from ex12 import generar_grafo, plotit

def arcs_from_cromosoma(cromo):
    return [(x,y) for x, y in zip(cromo, cromo[1:])]


def RandomPopulation(nodos, population_size):
    cromosoma = np.array(range(nodos.shape[0]))
    pcr = [ (np.random.shuffle(cromosoma), cromosoma.copy() )[1] for i in range(population_size) ]
    return pcr

def RankRoutes(nodos, population):
    ranking = []
    for p in population:
        d = 0
        for x, y in zip(p, p[1:]):
            d+= np.linalg.norm(nodos[x] - nodos[y])
        ranking.append((d,p))
    return ranking

def GetNextGeneration(parents, mutation_rate):
    return [p[1] for p in parents]

def PerformSelection(fitness_result, elite_size):
    print(elite_size)
    fitness_result.sort(key=operator.itemgetter(0))
    return fitness_result[:elite_size]

if __name__ == '__main__':

    # parametros
    semilla = int(os.getenv('SEMILLA', time.time_ns() % 2**32))
    cuantos = int(os.getenv('NODOS', 6))
    dimensiones = int(os.getenv('DIMENSIONES', 2))
    no_improvements_max = int(os.getenv('NO_IMPROVMENTS_MAX', 2))
    population_size = int(os.getenv('POPULATION_SIZE', 10))
    elite_size = int(os.getenv('ELITE_SIZE', 5))
    mutation_rate = float(os.getenv('MUTATION_RATE', 0.5))
    np.random.seed(semilla)                                     # Seteamos la semilla del random

    nodos = generar_grafo(cuantos, dimensiones)                 # Generamos al azar una serie de puntos en el espacio:
    initial_population = RandomPopulation(nodos, population_size)

    fitness_result = RankRoutes(nodos, initial_population)
    best_fit = fitness_result[0][0]

    no_improvements = 0
    while no_improvements_max > no_improvements:
        parents = PerformSelection(fitness_result, elite_size)
        population = GetNextGeneration(parents, mutation_rate)
        fitness_result = RankRoutes(nodos, population)

        if fitness_result[0][0] > best_fit:
            best_fit = fitness_result[0][0]
        else:
            no_improvements += 1

    arcos = arcs_from_cromosoma(fitness_result[0][1])
    print("El camino es: %s" % arcos)
    print(f"El costo total es: {best_fit}")
    plotit(nodos, arcos)