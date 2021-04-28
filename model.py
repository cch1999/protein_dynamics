import torch
import torch.nn as nn
from pykeops.torch import LazyTensor
from utils import MLP, MLP_with_layernorm, construct_graph, knn, rmsd
import matplotlib.pyplot as plt
import torch_geometric
import os



n_objects = 3
obj_dim = 5  # mass, x pos, y pos, x speed, y speed

n_relations = n_objects * (n_objects - 1)
rel_dim = 1

eff_dim = 100
hidden_obj_dim = 100
hidden_rel_dim = 100


class RelationModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RelationModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size),
			nn.ReLU()
		)

	def forward(self, x):
		'''
		Args:
			x: [n_relations, input_size]
		Returns:
			[n_relations, output_size]
		'''
		return self.model(x)


class ObjectModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(ObjectModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
		)

	def forward(self, x):
		'''
		Args:
			x: [n_objects, input_size]
		Returns:
			[n_objects, output_size]
		Note: output_size = number of states we want to predict
		'''
		return self.model(x)


class InteractionNetwork(nn.Module):
	def __init__(self, dim_obj, dim_rel, dim_eff, dim_hidden_obj, dim_hidden_rel, dim_x=0):
		super(InteractionNetwork, self).__init__()
		self.rm = RelationModel(dim_obj * 2 + dim_rel, dim_hidden_rel, dim_eff)
		self.om = ObjectModel(dim_obj + dim_eff + dim_x, dim_hidden_obj, 128)  # x, y

	def m(self, obj, rr, rs, ra):
		"""
		The marshalling function;
		computes the matrix products ORr and ORs and concatenates them with Ra
		:param obj: object states
		:param rr: receiver relations
		:param rs: sender relations
		:param ra: relation info
		:return:
		"""
		# TODO move convertion to float tensor earlier to avoid repeats

		orr = obj.t().mm(rr)   # (obj_dim, n_relations)
		ors = obj.t().mm(rs)   # (obj_dim, n_relations)

		return torch.cat([orr, ors, ra.t()])   # (obj_dim*2+rel_dim, n_relations)

	def forward(self, obj, rr, rs, ra, x=None):
		"""
		objects, sender_relations, receiver_relations, relation_info
		:param obj: (n_objects, obj_dim)
		:param rr: (n_objects, n_relations)
		:param rs: (n_objects, n_relations)
		:param ra: (n_relations, rel_dim)
		:param x: external forces, default to None
		:return:
		"""
		# marshalling function
		b = self.m(obj, rr, rs, ra)   # shape of b = (obj_dim*2+rel_dim, n_relations)

		# relation module
		e = self.rm(b.t())   # shape of e = (n_relations, eff_dim)
		e = e.t()   # shape of e = (eff_dim, n_relations)

		# effect aggregator
		if x is None:
			a = torch.cat([obj.t(), e.mm(rr.t())])   # shape of a = (obj_dim+eff_dim, n_objects)
		else:
			a = torch.cat([obj.t(), x, e.mm(rr.t())])   # shape of a = (obj_dim+ext_dim+eff_dim, n_objects)

		# object module
		p = self.om(a.t())   # shape of p = (n_objects, 2)

		return p, e.t()

class Simulator(torch.nn.Module):

	def __init__(self):
		super(Simulator, self).__init__()

		self.node_encoder = MLP(25, 128, 1, 128)
		self.edge_encoder = MLP(5, 128, 1, 128)
		self.interaction_net = nn.ModuleList()

		for i in range(5):
			self.interaction_net.append(InteractionNetwork(128, 128, 128, 128, 128))

		self.decoder = MLP(128, 128, 2, 3)

	def forward(self, P, k):

		# Init with random velocites and update positions
		P.vels = torch.randn(G.pos.shape) * 0.001
		print("No help RMSD:",
			rmsd(P.pos, P.pos + (P.vels*20))[0])
		
		P.pos = P.pos + P.vels

		for i in range(20):
			# Learned simulator
			P = self.encode(P, k)
			P = self.process(P)
			P = self.decode(P)
			
			# Update atom coordinates
			P = self.update(P)
		
		return P

	def encode(self, P, k):
		# Compute connectiviity and embed nodes and edges
		P.edge_index, P.edge_attr = knn(P, k)

		P.x = P.node_f.clone()

		n_atoms = P.x.shape[0]
		n_relations = P.edge_index.shape[0]

		senders = torch.zeros([n_atoms, n_relations])
		recievers = torch.zeros([n_atoms, n_relations])

		for i, (sr,rr) in enumerate(zip(P.edge_index[:,0], P.edge_index[:,1])):
			senders[sr,i] = 1
			recievers[rr,i] = 1


		P.senders = senders
		P.receivers = recievers

		# Embed node properties
		P.x = self.node_encoder(G.x)
		P.edge_attr = self.edge_encoder(P.edge_attr)

		return P

	def process(self, P):
		"""
		Interaction network?
		"""
		# Process latent graph
		# TODO add residual connection
		for i, layer in enumerate(self.interaction_net):
			residual_x, residual_edge = P.x, P.edge_attr
			P.x, P.edge_attr = layer(P.x, P.senders, P.receivers, P.edge_attr)
			P.x, P.edge_attr = P.x+residual_x, P.edge_attr+residual_edge
		return P

	def decode(self, P):
		"""
		Extract forces from nodes
		"""
		P.x = self.decoder(P.x)
		return P

	def update(self, P):
		"""
		Update coordinates
		"""
		acc = P.x#/P.mass[:, None]
	
		total_acc = acc.norm(dim=-1)

		"""
		print("Acceleration", total_acc.mean())
		print("Velocity", P.vels.norm(-1).mean())
		"""

		# Intergrator (assuming dT = 1)
		P.vels = P.vels + acc
		P.pos = P.pos + P.vels

		return P


model = Simulator()
optimizer = torch.optim.Adam(model.parameters())

model.train()
optimizer.zero_grad()

losses = []
test_losses = []

data_dir = "protein_data/train_val/"
data = os.listdir(data_dir)

for i in range(400):

	total_loss = 0.0

	for protein in data[:1]:
		G = construct_graph(data_dir+protein)
		native_coords = G.pos

		out = model(G, 15)
		loss, passed = rmsd(native_coords, out.pos)

		if passed:
			loss_log = torch.log(1.0 + loss)
			loss_log.backward(retain_graph=True)
			optimizer.step()
			optimizer.zero_grad()
			print(i)
			print(loss)
			total_loss += float(loss)
	
	losses.append(total_loss/10)
	"""
	# Test
	with torch.no_grad():
		G = construct_graph(data_dir+data[-1])
		native_coords = G.pos
		test_out = model(G, 15)
		test_loss, passed = rmsd(native_coords, test_out.pos)
		if passed:
			print("Test loss:", test_loss)
			test_losses.append(test_loss)
	"""
exit()
from cgdms import starting_coords

G = construct_graph(data_dir+data[0])
native_coords = G.pos

G.pos = starting_coords(G.seq)

print("Starting RMSD", rmsd(native_coords, G.pos)[0])

rmsds = []

for i in range(1000):
	G = model(G, 15)
	loss = rmsd(native_coords, G.pos)[0]
	print(loss)
	rmsds.append(loss)
"""
plt.plot(losses, label="Train")
plt.plot(test_losses, label="Test")
"""
plt.plot(rmsds)
plt.ylim(0)
plt.xlim(0)
plt.ylabel("Loss - RMSD (A)")
plt.xlabel("Epoch")
plt.legend()
plt.show()
