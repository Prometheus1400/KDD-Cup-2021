from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class CNN_Layer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CNN_Layer, self).__init__(aggr="add")
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.bond_encoder = BondEncoder(emb_dim=100)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        x = x.float()
        edge_embedding = self.bond_encoder(edge_attr)
        # print(x.shape)
        # print(edge_index.shape)
        # print(edge_attr.shape)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_embedding)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
