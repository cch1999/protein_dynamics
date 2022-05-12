import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch_geometric.nn import MessagePassing, knn_graph


def exists(val):
    return val is not None


class EGNN_Sparse(MessagePassing):
    """Different from the above since it separates the edge assignment
    from the computation (this allows for great reduction in time and
    computations when the graph is locally or sparse connected).
    * aggr: one of ["add", "mean", "max"]
    """

    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=16,
        soft_edge=0,
        norm_feats=False,
        norm_coors=False,
        norm_coors_scale_init=1e-2,
        update_feats=True,
        update_coors=True,
        dropout=0.0,
        aggr="add",
        **kwargs
    ):
        assert aggr in {
            "add",
            "sum",
            "max",
            "mean",
        }, "pool method must be a valid option"
        assert (
            update_feats or update_coors
        ), "you must update either features, coordinates, or both"
        kwargs.setdefault("aggr", aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)
        # model params
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None

        self.edge_input_dim = edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        self.edge_weight = (
            nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edge else None
        )

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1.
        self.node_norm = (
            torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        )
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )

        self.node_mlp = (
            nn.Sequential(
                nn.Linear(feats_dim + m_dim, feats_dim * 2),
                self.dropout,
                nn.SiLU(),
                nn.Linear(feats_dim * 2, feats_dim),
            )
            if update_feats
            else None
        )

        # COORS
        self.coors_mlp = (
            nn.Sequential(
                nn.Linear(m_dim, m_dim * 4),
                self.dropout,
                nn.SiLU(),
                nn.Linear(self.m_dim * 4, 1),
            )
            if update_coors
            else None
        )

        self.vels_mlp = (
            nn.Sequential(
                nn.Linear(feats_dim, feats_dim * 2),
                self.dropout,
                nn.SiLU(),
                nn.Linear(feats_dim * 2, 1),
            )
            if update_coors
            else None
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        feats,
        coors,
        edge_index,
        edge_attr=None,
        batch=None,
        angle_data=None,
        size=None,
    ) -> Tensor:
        """Inputs:
        * x: (n_points, d) where d is pos_dims + feat_dims
        * edge_index: (n_edges, 2)
        * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
        * batch: (n_points,) long tensor. specifies xloud belonging for each point
        * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
        * size: None
        """

        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out, mhat_i = self.propagate(
            edge_index,
            x=feats,
            edge_attr=edge_attr_feats,
            coors=coors,
            rel_coors=rel_coors,
            batch=batch,
        )
        return hidden_out, coors_out, mhat_i

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index, size=None, **kwargs):
        """The initial call to start propagating messages.
        Args:
        `edge_index` holds the indices of a general (sparse)
            assignment matrix of shape :obj:`[N, M]`.
        size (tuple, optional) if none, the size will be inferred
            and assumed to be quadratic.
        **kwargs: Any additional data which is needed to construct and
            aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        coor_wij = self.coors_mlp(m_ij)
        # clamp if arg is set
        if self.coor_weights_clamp_value:
            coor_weights_clamp_value = self.coor_weights_clamp_value
            coor_weights.clamp_(min=-clamp_value, max=clamp_value)

        # normalize if needed
        kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

        mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)

        # vel_wi = self.vels_mlp(kwargs["x"])

        # vels_out = vel_wi*kwargs["vels"] + mhat_i/1000

        coors_out = kwargs["coors"] + mhat_i

        # update feats if specified
        # weight the edges if arg is passed
        if self.soft_edge:
            m_ij = m_ij * self.edge_weight(m_ij)
        m_i = self.aggregate(m_ij, **aggr_kwargs)

        hidden_feats = (
            self.node_norm(kwargs["x"], kwargs["batch"])
            if self.node_norm
            else kwargs["x"]
        )
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = kwargs["x"] + hidden_out

        # return tuple
        return self.update((hidden_out, coors_out, mhat_i), **update_kwargs)

    def __repr__(self):
        dict_print = {}
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__)


if __name__ == "__main__":

    from dynamics.data.datasets.greener.datamodule import GreenerDataModule
    from dynamics.model.utils.geometric import knn

    dm = GreenerDataModule(
        "/home/cch57/projects/protein_dynamics/dynamics/data/datasets/greener", 1
    )
    val_loader = dm.val_dataloader()

    k = 20

    for P in val_loader:

        P.edge_index = knn_graph(P.pos, k)

        layer = EGNN_Sparse(24)

        h, pos, edge_index = P.x, P.pos, P.edge_index

        for i in range(3):
            h, pos = layer(h, pos, edge_index)
