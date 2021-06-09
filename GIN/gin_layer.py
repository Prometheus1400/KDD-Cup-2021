from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


class GIN_Layer(MessagePassing):
    """ Convolution / Message Passing Layer"""

    def __init__(self, emb_dim):
        super(GIN_Layer, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        # x = x.float()
        edge_embedding = self.bond_encoder(edge_attr)
        # Step 2: Linearly transform node feature matrix.
        x = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        # Step 4-5: Start propagating messages.
        return x

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_Node(torch.nn.Module):
    """ Generates Node Embedding """

    def __init__(self,conv_type,emb_dim,num_layers,dropout_ratio=0.4,residual=False,JK="last",):
        """ JK: Last, Sum """
        super(GNN_Node, self).__init__()
        self.conv_type = conv_type
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.residual = residual
        self.JK = JK
        self.atom_encoder = AtomEncoder(emb_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == "gin":
                self.convs.append(GIN_Layer(emb_dim))
            else:
                ValueError(f"Undefined GNN type called {conv_type}")
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched):
        """ recieves batched data which is decomposed """
        x, edge_index, edge_attr = batched.x, batched.edge_index, batched.edge_attr

        embedding_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):

            emb = self.convs[layer](embedding_list[layer], edge_index, edge_attr)
            emb = self.batch_norms[layer](emb)

            if layer == (self.num_layers - 1):
                emb = F.dropout(emb, self.dropout_ratio, training=self.training)
            else:
                emb = F.dropout(F.relu(emb), self.dropout_ratio, training=self.training)

            if self.residual:
                emb += embedding_list[layer]
            embedding_list.append(emb)
        
        if self.JK == "last":
            final = embedding_list[-1]
        elif self.JK == "sum":
            final = 0
            for emb in embedding_list:
                final += emb
        elif self.JK == "skip":
            final = embedding_list[0] + embedding_list[-1]
        else:
            ValueError(f"Invalid JK connetion: {self.JK}")
        
        return final
        
