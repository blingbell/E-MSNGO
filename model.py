import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.heterograph import DGLGraph
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.nn import GraphConv, AvgPooling, MaxPooling, SumPooling
import numpy as np


class MSNGO(nn.Module): 
    def __init__(self, in_dim, h_dim, out_dim, n_hidden_layer, n_prop_step, mlp_drop=0.0,
                 attn_heads=1, feat_drop=0.0, attn_drop=0.0, residual=True, share_weight=False):
        super(MSNGO, self).__init__()
        self.n_prop_step = n_prop_step
        self.dropout = mlp_drop
        self.linear = nn.Linear(in_dim, h_dim)
        #self.embed_layer = nn.EmbeddingBag(in_dim, h_dim, mode="sum", include_last_offset=True)
        self.embed_bias = nn.Parameter(torch.zeros(h_dim))

        self.mlp = MLP(h_dim, h_dim, h_dim, n_hidden_layer, mlp_drop)

        self.prop_layers = nn.ModuleList([PropagateLayer(h_dim, h_dim, attn_heads, feat_drop, attn_drop,
                                                         residual=residual, share_weight=share_weight) for _ in range(n_prop_step)])

        self.output_layer = nn.Linear(h_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_uniform_(self.embed_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, blocks, x, y=None):
        h = F.relu(self.linear(x) + self.embed_bias)
        h = self.mlp(h)
        for block, prop_layer in zip(blocks, self.prop_layers):
            h, y = prop_layer(block, h, y)
        h = self.output_layer(h)
        return h, y

    def inference(self, g, idx, x_seq, x_struct, y, batch_size, device): 
        self.eval()
        unique_idx = np.unique(idx)
        unique_idx = unique_idx.astype(np.int32)
        index_mapping = {idx: i for i, idx in enumerate(unique_idx)}
        idx = np.asarray([ index_mapping[idx] for idx in idx ])

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_prop_step)
        dataloader = dgl.dataloading.DataLoader(g, unique_idx, sampler,
                                     batch_size=int(batch_size/2), shuffle=False, num_workers=1,  drop_last=False)
        x_output_list = []
        y_output_list = []
        with dataloader.enable_cpu_affinity():
            for input_nodes, _, blocks in dataloader:
                blocks = [blk.to(device) for blk in blocks]
                batch_seq = torch.from_numpy(x_seq[input_nodes.numpy()]).float().to(device)
                batch_struct = torch.from_numpy(x_struct[input_nodes.numpy()]).float().to(device)
                batch_x = torch.cat((batch_seq, batch_struct), -1)
                batch_y = torch.from_numpy(y[input_nodes.numpy()]).float().to(device)
                batch_x_hat, batch_y_hat = self.forward(blocks, batch_x, batch_y)
                x_output_list.append(torch.sigmoid(batch_x_hat).cpu().detach().numpy())
                y_output_list.append(batch_y_hat.cpu().detach().numpy())
        
        x_output = np.vstack(x_output_list)[idx]
        y_output = np.vstack(y_output_list)[idx]

        return x_output, y_output

class PropagateLayer(nn.Module):
    def __init__(self, in_dim, out_dim, attn_heads, 
                 feat_drop=0., attn_drop=0.,
                 residual=True,
                 activation=F.elu,
                 share_weight=True):
        super(PropagateLayer, self).__init__()
        self.attn_heads = attn_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.share_weight = share_weight
        if share_weight:
            self.gat = GraphAttention(in_dim, out_dim, attn_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=residual)
        else:
            self.gat_p = GraphAttention(in_dim, out_dim, attn_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=residual)
            self.gat_s = GraphAttention(in_dim, out_dim, attn_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=residual)
        self.feat_drop = nn.Dropout(feat_drop)
        self.residual = residual
        self.activation = activation

    def forward(self, block: DGLGraph, x, y):
        with block.local_scope():

            #block_p = block['interaction']
            #block_s = block['similarity']
            block_p = block['ppi']
            block_s = block['sim']

            if self.share_weight:
                h_p, a_p = self.gat(block_p, x)
                h_s, a_s = self.gat(block_s, x)
            else:
                h_p, a_p = self.gat_p(block_p, x)
                h_s, a_s = self.gat_s(block_s, x)

            h = self.activation(h_p + h_s)
            h = self.feat_drop(h)

            # label propagate
            if y != None:
                block_p.edata['a'] = a_p
                block_s.edata['a'] = a_s
                dst_flag = block.dstdata['flag']
                y0 = y[:block.number_of_dst_nodes()][dst_flag]
                block_p.srcdata.update({'y': y})
                block_p.update_all(fn.u_mul_e('y', 'a', 'm'),
                                   fn.sum('m', 'y_hat'))
                y_hat_i = block_p.dstdata.pop('y_hat')
                block_s.srcdata.update({'y': y})
                block_s.update_all(fn.u_mul_e('y', 'a', 'm'),
                                   fn.sum('m', 'y_hat'))
                y_hat_s = block_s.dstdata.pop('y_hat')
                y_hat = F.normalize(y_hat_i + y_hat_s)
                y_hat[dst_flag] = y0
            else:
                y_hat = y

            return h, y_hat


class GraphAttention(nn.Module):
    def __init__(self,
                 in_feats, out_feats, num_heads,
                 feat_drop=0., attn_drop=0.,
                 negative_slope=0.2, residual=False, activation=None):
        super(GraphAttention, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.fc_src = nn.Linear(
            self._in_src_feats, out_feats * num_heads)
        self.fc_dst = nn.Linear(
            self._in_src_feats, out_feats * num_heads)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.constant_(self.fc_src.bias, 0)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.constant_(self.fc_dst.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, graph: DGLGraph, feat):
        with graph.local_scope():
            h_src = h_dst = self.feat_drop(feat)
            feat_src = self.fc_src(h_src).view(
                -1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_src).view(
                 -1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
                h_dst = h_dst[:graph.number_of_dst_nodes()]

            graph.srcdata.update({'el': feat_src}) # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')) # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2) # (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads,1)
            # feature propagation
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst.mean(1), graph.edata['a'].mean(1)

            
class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, 
                 num_layers, dropout, norm = 'layer'):
        super(MLP, self).__init__()

        self.norm = norm
        self.dropout = dropout

        self.layers = nn.ModuleList([])

        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_d) for _ in range(num_layers-1)])
        self.norms.append(nn.LayerNorm(output_d))

        self.reset_parameters()

    def reset_parameters(self):
        """reset mlp parameters using xavier_norm"""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0)

    def forward(self, x):
        """The forward pass of mlp."""

        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        return x    
class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs:int=3,
                 pool_ratio:float=0.5, dropout:float=0.5):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_convs
        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio))
        self.convpools = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hid_dim * 2, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim)
        self.lin3 = torch.nn.Linear(hid_dim, out_dim)

    def forward(self, graph:dgl.DGLGraph):
        feat = graph.ndata["feature"]
        final_readout = None

        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout
        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = self.lin3(feat)
        return feat
    
    def encode(self, graph:dgl.DGLGraph):
        feat = graph.ndata["feature"]
        final_readout = None

        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout
        
        return final_readout


class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper 
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio=0.5, conv_op=GraphConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer1 = GraphConv(in_dim, 1)
        self.score_layer2 = GraphConv(in_dim, 1)
        self.non_linearity = non_linearity
        self.allow_zero_in_degree = True 
    
    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor):

        score1 = self.score_layer1(graph, feature).squeeze()
        score2 = self.score_layer2(graph, feature).squeeze()
        score  = (score1+score2)/2
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm


class ConvPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """
    def __init__(self, in_dim:int, out_dim:int, pool_ratio=0.5):
        super(ConvPoolBlock, self).__init__()
        self.conv1 = GraphConv(in_dim, out_dim)
        self.conv2 = GraphConv(out_dim, out_dim)
        self.pool = SAGPool(out_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.sumpool = SumPooling()
        self.allow_zero_in_degree = True   
    
    def forward(self, graph, feature):
        out = F.relu(self.conv1(graph, feature))
        out = torch.reshape(out,(-1,512))
        out = F.relu(self.conv2(graph, out))
        out = torch.reshape(out,(-1,512))
        out = F.relu(self.conv2(graph, out))
        out = torch.reshape(out,(-1,512))
        graph, out, _ = self.pool(graph, out)
        g_out = torch.cat([self.maxpool(graph, out), self.maxpool(graph, out)], dim=-1)
        return graph, out, g_out 
    

def get_batch_id(num_nodes:torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.
    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)


def topk(x:torch.Tensor, ratio:float, batch_id:torch.Tensor, num_nodes:torch.Tensor):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.
    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.
    
    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    
    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + 
        i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k