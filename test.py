from utils import knn, construct_graph

G = construct_graph("protein_data/example/1CRN.txt")


print(G)

edges, dists = knn(G.pos, 10)

print(edges)
print(dists)

print(edges.shape)
print(dists.shape)

G.edge_index = edges
G.edge_attr = dists

print(G)