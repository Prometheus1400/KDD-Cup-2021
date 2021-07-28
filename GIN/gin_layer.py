from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from torch._C import device
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.nn.functional as F
import random
import numpy as np
import config


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

    def __init__(
        self,
        conv_type,
        emb_dim,
        num_layers,
        dropout_ratio=0.4,
        residual=False,
        JK="last",
        noisy_node=False,
    ):
        """ JK: Last, Sum """
        super(GNN_Node, self).__init__()
        self.conv_type = conv_type
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.sum_weight = torch.nn.Parameter(torch.Tensor([1] * (num_layers + 1)))
        self.dropout_ratio = dropout_ratio
        self.residual = residual
        self.JK = JK
        self.noisy_node = noisy_node
        # self.atom_encoder = AtomEncoder(emb_dim)
        self.atom_encoder = torch.nn.Sequential(
            torch.nn.Linear(12, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == "gin":
                self.convs.append(GIN_Layer(emb_dim))
            else:
                ValueError(f"Undefined GNN type called {conv_type}")
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        if self.noisy_node:
            self.atom_decoder = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, 118)
                # prediction for node label (atomic number)
            )

    def forward(self, batched):
        """ recieves batched data which is decomposed """
        x, edge_index, edge_attr = batched.x, batched.edge_index, batched.edge_attr
        x = x.float()
        if self.noisy_node and self.training:
            # labels = torch.zeros((len(x), 118), device=config.device, dtype=torch.long)
            labels = x[:, 0].long()  # hopefully the atomic number of each atom
            labels.subtract(1)
            # subtract 1 so that atomic number 1 corresponds to index 0. i.e. H -> [1, 0, 0,...,0]
            for i in range(len(x)):
                num = random.randint(0, 99)
                if num <= 5:
                    noise_arr = (torch.randint(low=0,high=4000,size=(1, 9),device=config.device,dtype=x.dtype,)[0] / 1000)
                    x[i] *= noise_arr

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
        elif self.JK == "weighted_sum":
            final = 0
            for i, emb in enumerate(embedding_list):
                final += emb * ((i + 1) / len(embedding_list))
        elif self.JK == "learnable_sum":
            final = 0
            for i, emb in enumerate(embedding_list):
                final += emb * self.sum_weight[i]
        elif self.JK == "skip":
            final = embedding_list[0] + embedding_list[-1]
        else:
            ValueError(f"Invalid JK connetion: {self.JK}")

        if self.noisy_node and self.training:
            return [final, self.atom_decoder(embedding_list[-1]), labels]
        return [final]


### Virtual GNN to generate node embedding
class GNN_Node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layers,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_type="gin",
    ):
        """
            emb_dim (int): node embedding dimensionality
        """

        super(GNN_Node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.sum_weight = torch.nn.Parameter(torch.Tensor([1] * (num_layers + 1)))
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == "gin":
                self.convs.append(GIN_Layer(emb_dim))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(),
                )
            )

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = (
                    global_add_pool(h_list[layer], batch) + virtualnode_embedding
                )
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in h_list:
                node_representation += layer
        elif self.JK == "learnable_sum":
            node_representation = 0
            for i, layer in enumerate(h_list):
                node_representation += layer * self.sum_weight[i]
        elif self.JK == "weighted_sum":
            node_representation = 0
            for i, layer in enumerate(h_list):
                node_representation += layer * (i + 1) / len(h_list)
                # fixed weighted sum giving more emphasis to more recent layers

        return node_representation
