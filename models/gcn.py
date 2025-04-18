# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = dropout
        
        # 参数初始化
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv1.lin.weight)
        nn.init.kaiming_normal_(self.conv2.lin.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze()
    
class MultiHopGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, hops=2):
        super(MultiHopGCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.hops = hops
        # 不同跳数的权重参数
        self.weights = nn.Parameter(torch.randn(hops))
        
    def forward(self, x, edge_index):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 计算度矩阵归一化系数
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 初始节点特征变换
        x = self.lin(x)
        
        # 多跳信息聚合
        out = torch.zeros_like(x)
        h = x  # 当前跳的节点表示
        for k in range(self.hops):
            # 消息传播
            h = self.propagate(edge_index, x=h, norm=norm)
            # 加权聚合
            out += self.weights[k] * h
        
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
class MultiHopGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, hops=2):
        super(MultiHopGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(MultiHopGCNConv(in_channels, hidden_channels, hops))
        for _ in range(num_layers - 2):
            self.convs.append(MultiHopGCNConv(hidden_channels, hidden_channels, hops))
        self.convs.append(MultiHopGCNConv(hidden_channels, out_channels, hops))
        
        self.reg_params = self.convs[0:-1].parameters()
        self.non_reg_params = self.convs[-1].parameters()
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x