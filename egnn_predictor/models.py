import torch
import torch.nn as nn
import numpy as np

from egnn_predictor.gcl import E_GCL, GCL
from utils.utils_edm import remove_mean, remove_mean_with_mask

class EGNN_predictor(nn.Module):

    """
    Static EGNN-based graph-level predictor.

    This model operates on a ring-level molecular graph (with optional
    heteroatom nodes). It produces a single prediction per graph by
    aggregating node embeddings after equivariant message passing.

    There is NO time dependency and NO diffusion dynamics.
    """

    def __init__(
        self,
        in_nf=1,
        out_nf=1,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        tanh=False,
        agg="sum",
        mean=None,
        std=None,
        d=3,
        coords_range=15,
    ):
        super().__init__()
        self.d = d
        self.device = device

        self.egnn = EGNN(
            in_node_nf=in_nf,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            out_node_nf=out_nf,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
            coords_range=coords_range,
        )

        self.mean = mean.to(device) if mean is not None else None
        self.std = std.to(device) if std is not None else None

        # Cache for fully-connected adjacency
        self._edges_dict = {}


    def forward(self, xh, node_mask, edge_mask):
        """
        Parameters
        ----------
        xh : Tensor [B, N, d + F]
            Node inputs. First `d` channels are coordinates,
            remaining channels are node features.
        node_mask : Tensor [B, N, 1]
            Mask indicating valid nodes.
        edge_mask : Tensor [B, N*N, 1]
            Mask indicating valid edges.

        Returns
        -------
        Tensor [B, out_nf]
            Graph-level prediction.
        """
        bs, n_nodes, dims = xh.shape
        
        # Build fully-connected graph
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [e.to(xh.device) for e in edges]
        
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        
        # Split coordinates and node features
        x = xh[:, :, :self.d].reshape(bs * n_nodes, self.d) * node_mask
        h = xh[:, :, self.d:].reshape(bs * n_nodes, -1) * node_mask
        
        # x = xh[:, :, : self.d].view(bs * n_nodes, -1).clone() * node_mask
        # h = xh[:, :, self.d :].view(bs * n_nodes, -1).clone() * node_mask

        # Edge attributes: squared distance
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        
        # EGNN message passing
        h_final, _ = self.egnn(
            h, x, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=edge_attr
        )
        
        # Graph-level aggregation
        h_final = h_final.view(bs, n_nodes, -1)
        return h_final.mean(dim=1)

    # def get_adj_matrix(self, n_nodes, batch_size, device):
    #     if n_nodes in self._edges_dict:
    #         edges_dic_b = self._edges_dict[n_nodes]
    #         if batch_size in edges_dic_b:
    #             return edges_dic_b[batch_size]
    #         else:
    #             # get edges for a single sample
    #             rows, cols = [], []
    #             for batch_idx in range(batch_size):
    #                 for i in range(n_nodes):
    #                     for j in range(n_nodes):
    #                         rows.append(i + batch_idx * n_nodes)
    #                         cols.append(j + batch_idx * n_nodes)
    #             edges = [
    #                 torch.LongTensor(rows).to(device),
    #                 torch.LongTensor(cols).to(device),
    #             ]
    #             edges_dic_b[batch_size] = edges
    #     else:
    #         self._edges_dict[n_nodes] = {}
    #         return self.get_adj_matrix(n_nodes, batch_size, device)

    #     return edges

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes not in self._edges_dict:
            self._edges_dict[n_nodes] = {}

        if batch_size not in self._edges_dict[n_nodes]:
            rows, cols = [], []
            for b in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + b * n_nodes)
                        cols.append(j + b * n_nodes)

            self._edges_dict[n_nodes][batch_size] = [
                torch.LongTensor(rows).to(device),
                torch.LongTensor(cols).to(device),
            ]

        return self._edges_dict[n_nodes][batch_size]

    def unwrap_forward(self):
        return self._forward

    def unnormalize(self, pred):
        if self.mean is not None:
            pred = pred * self.std + self.mean
        return pred


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
    ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        # self.reg = reg
        ### Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                ),
            )

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](
                h,
                edges,
                x,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        out_node_nf=None,
    ):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                ),
            )

        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h
