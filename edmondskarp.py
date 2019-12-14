import numpy as np

def EdmondsKarp(graph, s, t):
    g = graph.copy()
    maxflow = 0
    path = [-1]*graph.shape[0]
    pathExists = True
    print("Running Edmonds Karp algorithm")
    while BFS(g,s, t, path):
        c = np.Inf
        v = t
        while v != s:
            c = min(c, g[path[v]][v])
            v = path[v]

        maxflow = maxflow + c
        v = t
        while v != s:
            u = path[v]
            g[u][v] = g[u][v] - c
            g[v][u] = g[v][u] + c
            v = path[v]

    found = [-1]*graph.shape[0]
    DFS(g,s, found)

    cuts = []
    for i in range(g.shape[0]):
        for j in range(g.shape[0]):
            if found[i] == 1 and found[j] == -1 and graph[i][j]:
                cuts.append((i, j))
    return cuts



def DFS(graph, s, found):

    stack = [s]
    while stack:
        u = stack.pop()
        if found[u] == -1:
            found[u] = 1
            stack.extend([v for v in range(graph.shape[0]) if graph[u][v]])


def BFS(graph, s, t, path):
    Q = []
    visited = np.zeros(graph.shape[0], dtype=bool)
    Q.append(s)
    visited[s] = True
    path[s]  = -1

    while len(Q)!=0:
        u = Q.pop(0)
        for v in range(graph.shape[0]):
            if not(visited[v]) and graph[u][v] > 0:
                Q.append(v)
                path[v] = u
                visited[v] = True
    return visited[v]
