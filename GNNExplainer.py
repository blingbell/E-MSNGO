import torch
from tqdm import tqdm
import dgl

EPS = 1e-15


class GNNExplainer(torch.nn.Module):
    """ GNNExplainer for MSNGO, Supporting DGL Blocks-Sampled Subgraphs """

    coeffs = {
        'edge_size': 0.01,
        'edge_reduction': 'mean',
        'node_feat_size': 0.5,
        'node_feat_reduction': 'mean',
        'edge_ent': 0.5,
        'node_feat_ent': 0.2,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log

    def __set_masks__(self, x, num_edges):
        F = x.size(1)  
        self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * 0.1)
        self.edge_mask = torch.nn.Parameter(torch.randn(num_edges) * 0.1)

    def __clear_masks__(self):
        self.node_feat_mask = None
        self.edge_mask = None

    def __loss__(self, log_logits, pred_label):
        probs = torch.sigmoid(log_logits) 
    
        # Binary Cross-Entropy Loss
        loss_fn = torch.nn.BCELoss()
        target_labels = pred_label.float()  
        loss = loss_fn(probs, target_labels)  #  BCE loss

        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + 1e-15) - (1 - m) * torch.log(1 - m + 1e-15)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + 1e-15) - (1 - m) * torch.log(1 - m + 1e-15)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss


    def apply_edge_mask(self, blocks, edge_mask, etype):
        device = blocks[0].device
        mask = (edge_mask > 0.5).to(device)
        edge_size = [0]
        mask_blocks = []
        
        for block in blocks:
            edge_size.append(block.edges(etype=etype)[0].size(0))

        ntype = blocks[0].ntypes[0]
        for i, subgraph in enumerate(blocks):
            canonical_etypes = subgraph.canonical_etypes
            all_src, all_dst = [], []
            all_weights = {}
            preserved_nodes = set()
            
            for e in canonical_etypes:
                src, dst = subgraph.edges(etype=e)
                weights = subgraph.edges[e].data['w']  # get weight
                
                if e[1] == etype:  # target edge type
                    # comput mask limit
                    start_idx = edge_size[i]
                    end_idx = edge_size[i] + edge_size[i+1]
                    current_mask = mask[start_idx:end_idx]

                    # apply mask
                    filtered_src = src[current_mask]
                    filtered_dst = dst[current_mask]

                    if filtered_src.numel() == 0 :
                        filtered_src = filtered_dst = torch.tensor([0], device=device)
                        all_weights[etype] = torch.tensor([0.0], device=device)
                    else:
                        all_weights[etype]=weights[current_mask]

                    all_src.append(filtered_src)
                    all_dst.append(filtered_dst)
                    
                    preserved_nodes.update(filtered_src.tolist())
                    preserved_nodes.update(filtered_dst.tolist())
                else:  
                    if src.numel() == 0 :
                        src = dst = torch.tensor([0], device=device)
                        all_weights[e[1]] = torch.tensor([0.0], device=device)
                    else:
                        all_weights[e[1]]=weights
                    all_src.append(src)
                    all_dst.append(dst)
                    preserved_nodes.update(src.tolist())
                    preserved_nodes.update(dst.tolist())

            preserved_nodes = sorted(preserved_nodes)
            node_mapping = {n: i for i, n in enumerate(preserved_nodes)}
            
            remapped_src = []
            remapped_dst = []

            for src, dst in zip(all_src, all_dst):
                remapped_src.append(torch.tensor([node_mapping[n.item()] for n in src], device=device))
                remapped_dst.append(torch.tensor([node_mapping[n.item()] for n in dst], device=device))
  
            for i in range(len(remapped_src)):
                if remapped_src[i].numel() == 0:
                    remapped_src[i] = remapped_dst[i]
                    if remapped_src[i].numel() == 0:
                        print('remapped_src[i] = remapped_dst[i] = ',remapped_src[i])
            new_block = dgl.to_block(
            dgl.heterograph(
                {e: (remapped_src[i], remapped_dst[i]) for i, e in enumerate(canonical_etypes)},
                num_nodes_dict={ntype: len(preserved_nodes)}
            ),
            include_dst_in_src=True
        ).to(device)

            new_block['ppi'].edata['w'] = all_weights['ppi']
            new_block['sim'].edata['w'] = all_weights['sim']

            original_nids = subgraph.srcnodes[ntype].data[dgl.NID]
            original_flags = subgraph.srcnodes[ntype].data['flag']
            new_nids = torch.tensor([original_nids[n] for n in preserved_nodes], device=device)
            new_flags = torch.tensor([original_flags[n] for n in preserved_nodes], device=device)  
            new_block.srcnodes[ntype].data[dgl.NID] = new_nids
            new_block.srcnodes[ntype].data['flag'] = new_flags  

            dst_nids = new_block.dstnodes[ntype].data[dgl.NID]
            src_nids = new_block.srcnodes[ntype].data[dgl.NID]

            src_nids_set = {i:n.item() for i, n in enumerate(src_nids)}
            dst_indices = torch.tensor([src_nids_set[i.item()] for i in dst_nids], device=device)

            new_block.dstnodes[ntype].data[dgl.NID] = dst_indices
            new_block.dstnodes[ntype].data['flag'] = new_flags[torch.tensor([0])]

            mask_blocks.append(new_block)
        original_nodes = blocks[0].srcnodes[ntype].data[dgl.NID]
        new_nodes = mask_blocks[0].srcnodes[ntype].data[dgl.NID]
        mask_node = torch.isin(original_nodes, new_nodes)

        return mask_blocks, mask_node

    def explain_node(self, node_idx, blocks, x, label, etype="ppi"):
        """
        Explain the PPI or SIM relationships of a given node.  

        - node_idx: Target node  
        - blocks: Subgraph sampled by DGL  
        - x: Node features of the subgraph  
        - etype: Edge type to analyze ("ppi" or "sim")
        """
        self.model.eval()
        self.__clear_masks__()
        num_edges = 0
        for block in blocks:
            edge_index = block.edges(etype=etype)
            num_edges += edge_index[0].size(0)

        with torch.no_grad():
            out, _ = self.model(blocks, x, label)
            pred_label = torch.sigmoid(out)

        # initial mask
        self.__set_masks__(x, num_edges)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.epochs, desc=f"explain node {node_idx}")

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            # apply feature mask
            h = x * self.node_feat_mask.sigmoid()
            y = label
            # apply edges mask
            masked_blocks, mask = self.apply_edge_mask(blocks, self.edge_mask.sigmoid(), etype)
            h = h[mask,:]
            if label:
                y = y[mask,:]
            log_logits, _ = self.model(masked_blocks, h, y)
            loss = self.__loss__(log_logits, pred_label)

            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        # obtain final mask
        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask