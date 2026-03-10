import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from model.MACCS import MACCSEmbedder
from torch_geometric.nn import MessagePassing
import torch_scatter
from torch_geometric import nn as pyg_nn
from torch_geometric.utils import softmax
import torch_geometric

class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=2, dropout=0.2, negative_slope=0.2):
        super().__init__(node_dim=0)
        self.heads = heads
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)
        alpha_l = (x * self.att_l).sum(dim=-1)
        alpha_r = (x * self.att_r).sum(dim=-1)
        alpha = (alpha_l, alpha_r)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.view(-1, H * C) + self.bias  # bias added after concatenation
        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, self.negative_slope)
        alpha = torch_geometric.utils.softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(inputs, index, dim=0, reduce='sum')


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, args, emb=False, output_dim = 1, use_maccs=True):
        super().__init__()
        self.use_maccs = use_maccs
        self.emb = emb
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.heads = args.heads

        self.maccs_proj = MACCSEmbedder(167, 8) if use_maccs else None

        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        # self.convs.append(conv_model(input_dim, hidden_dim, heads=self.heads))
        self.convs.append(GATConv(input_dim, hidden_dim, heads=self.heads, dropout=self.dropout))


        for _ in range(self.num_layers - 1):
            # self.convs.append(conv_model(self.heads * hidden_dim, hidden_dim, heads=self.heads))
            self.convs.append(GATConv(hidden_dim * self.heads, hidden_dim, heads=self.heads, dropout=self.dropout))

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * self.heads) for _ in range(self.num_layers)
        ])

        gnn_out_dim = self.heads * hidden_dim
        if use_maccs:
            aux_dim = 2 + (8 if use_maccs else 167)
        else:
            aux_dim = 2
        total_input_dim = gnn_out_dim + aux_dim

        self.post_mp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GATLayer
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)               # ← Add this line
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = pyg_nn.global_mean_pool(x, batch)

        graph_features = []
        if hasattr(data, "MW"):
            graph_features.append(data.MW.view(-1, 1))
        if hasattr(data, "ALOGP"):
            graph_features.append(data.ALOGP.view(-1, 1))

        if hasattr(data, "MACCS"):
            maccs = data.MACCS #.view(-1, 167)
            if self.use_maccs and self.maccs_proj is not None:
                maccs = self.maccs_proj(maccs)
            graph_features.append(maccs)

        if graph_features:
            x = torch.cat([x] + graph_features, dim=1)

        return x if self.emb else self.post_mp(x)

    def loss(self, pred, label):
        return F.binary_cross_entropy_with_logits(pred.view(-1), label.float())
