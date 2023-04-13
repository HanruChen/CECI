import math
import numpy as np
from itertools import combinations, permutations
from CI import CI_I, CI_II, getCorr
from evaluation import *
from IGCI import igci


def has_path(num_nodes, adj_, source, dest):
    queue = np.zeros([num_nodes]).astype('int')
    vis = np.zeros([num_nodes]).astype('bool')
    queue_l = 0
    queue_r = 1
    queue[0] = source
    vis[source] = True

    while queue_l < queue_r:
        cur = queue[queue_l]
        queue_l += 1
        if cur == dest:
            return True
        for nxt in range(num_nodes):
            if (vis[nxt] == True):
                continue
            if not (adj_[cur][nxt] == 1 and adj_[nxt][cur] == 0):
                continue
            queue[queue_r] = nxt
            queue_r += 1
            vis[nxt] = True

    return False


def CECI(data, max_size, alpha, T1, T2):
    num_nodes = len(data)
    # prepare a complete undirected graph
    adj = np.zeros([num_nodes, num_nodes]).astype('int')
    for (x, y) in permutations(range(num_nodes), 2):
        adj[x][y] = 1
    sep_set = [[tuple() for j in range(num_nodes)] for i in range(num_nodes)]

    # calc skeleton
    for size in range(max_size):
        for (x, y) in permutations(range(num_nodes), 2):
            if (adj[x][y] == 0):
                continue
            temp = []
            for z in range(num_nodes):
                if (adj[x][z] == 1 and z != y):
                    temp.append(z)
            temp = tuple(temp)
            for sep in combinations(temp, size):
                isCI_II = CI_II(data, max_size, x, y, sep, 0.01)
                if isCI_II:
                    isCI_I = CI_I(data, x, y, sep, alpha)

                    if isCI_I:
                        adj[x][y] = 0
                        adj[y][x] = 0
                        sep_set[x][y] += sep
                        break

    # identify V-structure
    for (x, z) in combinations(range(num_nodes), 2):
        if adj[x][z] != 1 or adj[z][x] != 1:
            continue
        for y in range(num_nodes):
            if x == y or z == y:
                continue
            if adj[x][y] != 0 or adj[y][x] != 0 or adj[y][z] != 1 or adj[z][y] != 1:
                continue
            if not z in sep_set[x][y]:
                adj[z][x] = adj[z][y] = 0

    # Transport the direction
    while True:
        change_bool = False

        for (x, y) in permutations(range(num_nodes), 2):
            if not (adj[x][y] == 1 and adj[y][x] == 0):
                continue
            for z in set(range(num_nodes)) - set([x, y]):
                if not (adj[y][z] == 1 and adj[z][y] == 1):
                    continue
                if not (adj[x][z] == 0 and adj[z][x] == 0):
                    continue
                adj[z][y] = 0
                change_bool = True

        for (x, y) in permutations(range(num_nodes), 2):
            if not has_path(num_nodes, adj, x, y):
                continue
            if not (adj[x][y] == 1 and adj[y][x] == 1):
                continue
            adj[y][x] = 0
            change_bool = True

        if change_bool == False:
            break

    # IGCI
    for (i, j) in combinations(range(num_nodes), 2):
        if adj[i][j] or adj[j][i]:
            value = igci(data[i].reshape(-1, 1), data[j].reshape(-1, 1))

            if math.fabs(value) < T1:
                adj[i][j] = adj[j][i] = 1
            elif math.fabs(value) > T2:
                if value < 0:
                    adj[i][j] = 1
                    adj[j][i] = 0
                else:
                    adj[j][i] = 1
                    adj[i][j] = 0

    return adj

