import networkx as nx


def plot_network(graph, classes_disp):
    G = nx.Graph()
    V, E = [], []
    for k, v in graph.items():
        V.append(k)
        for i in v:
            if len(v) == 0: continue
            E.append((k, i))

    labels = dict()
    for i in V: labels[i] = \
        classes_disp[i]

    for v in V:
        G.add_node(labels[i])

    for (u,v) in E:
        G.add_edge(labels[u], labels[v])

    nx.write_graphml(G, 'Chow-Liu.graphml')




