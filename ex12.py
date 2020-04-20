"""

Traveling Salesman Problem
#EX12: LP with subtours
1) Try to solve the TSP with a LP matching model. Use the scipy.linprog package. For the cities coordinates use random points.
2) Plot the resulting network with the matplotlib library.
3) Understand the outputs. What happend to the
optimal path?
4) What are the alternatives to this formulation and
their disadvantages?

"""
import time

import networkx as nx

"""
G_1 = nx.Graph()
tempedgelist =  [[0, 2], [0, 3], [1, 2], [1, 4], [5, 3]]
G_1.add_edges_from(tempedgelist)

n_nodes = 6
pos = {i:(random.randint(0,50),random.randint(0,100)) for i in range(n_nodes)}
nx.draw(G_1, pos, edge_labels=True)
plt.show()

exit(0)
"""
import os
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def nn2na(NN):
  idx = np.argwhere(NN)
  NA = np.zeros([NN.shape[0], idx.shape[0]]).astype(int)
  for i, arc in enumerate(idx):
    NA[arc[0], i] = 1
    NA[arc[1], i] = -1

  arc_idx = [ (arc[0], arc[1]) for arc in idx]
  return NA, arc_idx

def get_usage(arc_idxs, use, max_flow):
  return [f"{x} -> {np.round(use[i])} / {max_flow[i]}" for i, x in enumerate(arc_idxs)]

def min_cut(arc_idxs, use, max_flow):
  return list(filter(lambda x: x is not None,
                     [x if max_flow[i] != None and np.isclose(use[i], max_flow[i]) == [True] else None for i, x in
                      enumerate(arc_idxs)]))

def get_selected_arcs(arc_idxs, selected_arcs):
  arc = []
  for idx, i in enumerate(selected_arcs):
      if round(i) == 1:
          arc.append(arc_idxs[idx])
  return arc

def get_arcs_as_tuple_list(NN):
    return [ tuple(x) for x in (np.transpose(np.nonzero(NN)).tolist())]


def generar_grafo(nodos, dims = 2):
    return np.random.rand(nodos, dims)

def plotit(vertices, arcos):
    G_1 = nx.Graph()

    G_1.add_edges_from(arcos)

    pos = {i: (n[0]*100, n[1]*100) for i,n in enumerate( vertices ) }
    nx.draw(G_1, pos, edge_labels=True,with_labels=[ str(chr(97+i)) for i,n in enumerate(vertices) ])
    plt.show()


if __name__ == '__main__':

    cuantos = int(os.getenv('NODOS', 6))
    dimensiones = int(os.getenv('DIMENSIONES', 2))
    np.random.seed(time.time_ns() % 2**32)                 # Seteamos la semilla del random

    nodos = generar_grafo(cuantos, dimensiones)         # Generamos al azar una serie de puntos en el espacio:

    NN = np.zeros((cuantos,cuantos))
    C = []
    for i,p in enumerate(nodos):
        for j,q in enumerate(nodos):
            if i!=j:
                d = np.linalg.norm(p-q)
                C.append(d)
                NN[i][j] = 1

    Aeq1, arc_idxs = nn2na(NN)
    Aeq2 = Aeq1.copy()*-1
    Aeq1[Aeq1 < 0] = 0
    Aeq2[Aeq2 < 0] = 0

    beq = np.ones(cuantos*2, int)
    l = np.zeros(Aeq1.shape[1])
    u = np.full(Aeq1.shape[1], None)
    bounds = tuple(zip(l, u))

    Aeq = np.block([
        [Aeq1],
        [ Aeq2]
    ])

    res = linprog(C, A_eq=Aeq, b_eq=beq, bounds=bounds, method='revised simplex')

    print(res)
    print("El camino es: %s" % get_selected_arcs(arc_idxs, res.x))
    plotit(nodos ,get_selected_arcs(arc_idxs, res.x))
