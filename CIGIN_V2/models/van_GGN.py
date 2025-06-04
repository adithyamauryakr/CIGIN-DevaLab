import numpy as np
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set, NNConv, GATConv, GatedGraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatherModel(nn.Module):
    def __init__(self, node_input_dim=42, node_hidden_dim=42, num_heads=2, num_step_message_passing=6):

        super().__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.conv = GatedGraphConv(in_feats=node_hidden_dim,
                                   out_feats=node_hidden_dim,
                                   n_steps=num_step_message_passing,
                                   n_etypes=1)

        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)

    def forward(self, g, n_feat, e_feat=None):  # e_feat is unused in GATv2
        g = dgl.add_self_loop(g)
        init = n_feat.clone()
        h = F.relu(self.lin0(n_feat))
        edge_types = torch.zeros(g.num_edges(), dtype=torch.long, device=h.device)
        # If no edge type info, treat all edges the same
        out = self.conv(g, h, torch.zeros(g.num_edges(), dtype=torch.long, device=h.device))

        # Optional extra layer to mix updated & original features
        out = self.message_layer(torch.cat([out, h], dim=1))
        return out + init


class CIGINGGN(nn.Module):
    """
    This the main class for CIGIN model
    """

    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 interaction='dot',
                 num_step_set2_set=2,
                 num_layer_set2set=1,
                 ):
        super(CIGINGGN, self).__init__()
        init = n_feat.clone()
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.solute_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                         self.node_hidden_dim, self.edge_input_dim,
                                         self.num_step_message_passing,
                                         )
        self.solvent_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                          self.node_hidden_dim, self.edge_input_dim,
                                          self.num_step_message_passing,
                                          )
        # These three are the FFNN for prediction phase
        self.fc1 = nn.Linear(8 * self.node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_solute = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

    def forward(self, data):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        # node embeddings after interaction phase
        solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        try:
            # if edge exists in a molecule
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())
        except:
            # if edge doesn't exist in a molecule, for example in case of water
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), None)

        # Interaction phase
        len_map = torch.mm(solute_len.t(), solvent_len)

        if 'dot' not in self.interaction:
            X1 = solute_features.unsqueeze(0)
            Y1 = solvent_features.unsqueeze(1)
            X2 = X1.repeat(solvent_features.shape[0], 1, 1)
            Y2 = Y1.repeat(1, solute_features.shape[0], 1)
            Z = torch.cat([X2, Y2], -1)

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2)
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t())
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map = torch.mm(solute_features, solvent_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.node_hidden_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map)
            interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map)

        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        # Prediction phase
        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)

        solute_features = self.set2set_solute(solute, solute_features)
        solvent_features = self.set2set_solvent(solvent, solvent_features)

        final_features = torch.cat((solute_features, solvent_features), 1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        return predictions, ret_interaction_map
