import operator
import os
import time

import numpy as np
import matplotlib.pyplot as plt

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
    ranking.sort(key=operator.itemgetter(0))
    return ranking

def mutate(p):
    sw1 = np.random.randint(p.shape[0])
    sw2 = np.random.randint(p.shape[0])
    p[[sw1, sw2]] = p[[sw2, sw1]]
    return p

def cross_over(p, q):
    sw1 = np.random.randint(p.shape[0] - 1)
    sw2 = max(np.random.randint(p.shape[0] - 1 - sw1), 1)  # me aseguro de que no se pase
    r = np.full(p.shape[0], None)
    # copiamos los elementos
    for i in range(sw1, sw1 + sw2 + 1):
        r[i] = p[i]
    # ahora copiamos los elementos de q en r
    vacios_en_r = np.where(r == None)[0]
    j = 0
    for qi in q:
        if qi in r:
            continue
        else:
            r[vacios_en_r[j]] = qi
            j += 1
    return r


def GetNextGeneration(elite, parents, population_size, mutation_rate):
    rs = []
    ps = np.array([p[1] for p in elite])
    parents_size = len(parents)
    for i in range(population_size - len(parents)):
        # crossover
        idx = np.random.randint(parents_size)
        idx2 = np.random.randint(parents_size - idx) - 1
        p = parents[idx].copy()
        q = parents[idx2].copy()
        r = cross_over(p, q)
        rs.append(r)

    # mutates before sending back
    rs = [ r if np.random.rand() > mutation_rate else mutate(r) for r in rs ]
    for p in ps:
        rs.append(p)
    return rs

def PerformSelection(fitness_result, elite_size):
    # Fitness proportionate selection
    # obtengo la suma
    suma = sum([r[0] for r in fitness_result])
    # por cada elemento obtengo el (1 - peso/suma), un número que es más alto mientras mejor sea el cromosoma
    # luego lo multiplico por un factor random, que sería el paso de la ruleta.
    # luego los multiplico entre sí y ordeno por este facto. Tomos los 5 primeros casos.
    probas = [ (np.random.random() * (1-(r[0]/suma)),r) for r in fitness_result]

    probas.sort(key=operator.itemgetter(0))
    seleccion = [ p[1] for p in probas ][:elite_size] # rescato los fitness
    seleccion.sort(key=operator.itemgetter(0))
    seleccion = [ p[1] for p in seleccion ]
    #print(elite_size)
    fitness_result.sort(key=operator.itemgetter(0))
    #return fitness_result[:elite_size]
    return fitness_result[:elite_size], seleccion

if __name__ == '__main__':
    # parametros
    semilla = int(os.getenv(                        'SEMILLA', time.time_ns() % 2**32))
    cuantos = int(os.getenv(                          'NODOS',                     10))
    dimensiones = int(os.getenv(                'DIMENSIONES',                      2))
    no_improvements_max = int(os.getenv( 'NO_IMPROVMENTS_MAX',                     20))
    population_size = int(os.getenv(        'POPULATION_SIZE',                    100))
    elite_size = int(os.getenv(                  'ELITE_SIZE',                     10))
    mutation_rate = float(os.getenv(          'MUTATION_RATE',                   0.10))
    np.random.seed(semilla)                                     # Seteamos la semilla del random

    nodos = generar_grafo(cuantos, dimensiones)                 # Generamos al azar una serie de puntos en el espacio:
    population = RandomPopulation(nodos, population_size)

    fitness_result = RankRoutes(nodos, population)
    best_fit = fitness_result[0][0]

    no_improvements = 0
    fits = [best_fit]
    while no_improvements_max > no_improvements:
        elite, parents = PerformSelection(fitness_result, elite_size)
        population = GetNextGeneration(elite, parents, population_size, mutation_rate)
        fitness_result = RankRoutes(nodos, population)

        current_fit = fitness_result[0][0]
        fits.append(current_fit)

        if  current_fit > best_fit:
            best_fit = current_fit
            no_improvements = 0
        else:
            no_improvements += 1

    arcos = arcs_from_cromosoma(fitness_result[0][1])
    print("Mejor fit por generación: %s"% fits)
    print("El camino es: %s" % arcos)
    print(f"El costo total es: {best_fit}")
    print("Se obsera q es muy inestable")

    plt.xlabel('Generacion')
    plt.ylabel('Costo del recorrido')
    plt.plot( fits )
    plt.show()

    plotit(nodos, arcos)

