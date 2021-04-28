from utils import construct_graph_networkx
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

radii = np.arange(0,10,1)

triangles = []

G = construct_graph_networkx("protein_data/example/1CRN.txt", 8 ,False)

print(max(nx.algorithms.clique.find_cliques(G), key = len))
print(sum(list(nx.triangles(G).values()))/3)

data_dir = "protein_data/train_val/"
data = os.listdir(data_dir)

for radius in tqdm(radii):

    triangles_stats = []

    for prot in data:

        G = construct_graph_networkx(data_dir+prot, radius ,False)

        try:
            triangles_stats.append(len(G.edges))
        except:
            triangles_stats.append(0)

    median = np.median(np.array(triangles_stats))

    triangles.append(median)

plt.plot(radii, triangles)
plt.xlim(0)
plt.ylim(0)
plt.ylabel("Median no. of edges in protein graph)")
plt.xlabel("Connectivity radius (Å)")
plt.title("Median no. of edges in a protein graph (2000 proteins)")
plt.show()