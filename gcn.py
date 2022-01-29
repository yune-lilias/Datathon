import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import ASAPooling
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F


def readdata():
    doc2vec_model = Doc2Vec.load("test_doc2vec.model")
    word_feature_array = []
    for i in range(len(doc2vec_model.docvecs)):
        word_feature_array.append(doc2vec_model.docvecs[i])
    word_feature_array = np.array(word_feature_array)
    node_type = pd.read_csv("./datasets/Training/node_classification.csv")
    node_type_onehot = np.zeros([len(node_type), 4])
    for i in range(len(node_type)):
        node_type_onehot[i][node_type.iloc[i][1]-1] = 1
    all_feature_array = np.concatenate((word_feature_array, node_type_onehot), axis=1)
    edge = pd.read_csv("./datasets/Training/training_graph.csv")
    edge = edge.to_numpy().T
    all_data = Data(x=torch.tensor(all_feature_array, dtype=torch.float32),
        edge_index=torch.tensor(edge, dtype=torch.long), edge_attr=None)
    all_data.num_nodes = len(all_feature_array)
    all_data.num_features = len(all_feature_array[0])
    data = train_test_split_edges(all_data)
    return data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.data = readdata()
        self.device = 'cpu'
        self.data= self.data.to(self.device)


        self.conv1 = GCNConv(self.data.num_features, 256)
        self.nl = BatchNorm(256)
        #self.conv2 = GCNConv(256, 128)
        #self.conv2 = GCNConv(data.num_features, 128)
        self.conv2 = GATConv(self.data.num_features, 128)
        self.nl2 = BatchNorm(128)
        self.conv3 = GCNConv(128,64)

    def encode(self):
       # x = self.conv1(data.x, data.train_pos_edge_index)
       # x = x.relu()
        #x = self.nl(x)
        x = self.conv2(self.data.x, self.data.train_pos_edge_index)
        x = x.tanh()
        #x = self.nl2(x)
        return self.conv3(x, self.data.train_pos_edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def edgepred(self, z, te):
        prob_adj = z @ z.t()
        for k in range(len(te)):
            probability = prob_adj[te[k][0]][te[k][1]]
            probability = probability.detach().numpy()
            te[k].append(probability)
        return te

    def pred_one_edge(self, z, new_node, top_k=5, remove_negative=True):
        prob_adj = z @ z.t()
        rank_list = np.zeros(self.data.num_nodes)
        for k in range(self.data.num_nodes):
            if k == new_node:
                rank_list[k] = -np.inf
            else:
                probability = prob_adj[new_node][k]
                probability = probability.detach().numpy()
                rank_list[k] = probability

        index_list = rank_list.argsort()
        top_k_index = index_list[-1:-top_k - 2:-1]
        top_k_score = rank_list[top_k_index]

        if remove_negative:
            for neg_point, inde in enumerate(top_k_score):
                if inde < 0:
                    break
            top_k_index = top_k_index[:neg_point]
            top_k_score = top_k_score[:neg_point]

        return top_k_index, top_k_score

    def get_link_labels(self,pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=self.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def modelin(self,op):
        self.optimizer = op

    def trains(self):
        self.train()
        neg_edge_index = negative_sampling(
            edge_index=self.data.train_pos_edge_index, num_nodes=self.data.num_nodes,
            num_neg_samples=self.data.train_pos_edge_index.size(1),
            force_undirected=True,
        )
        self.optimizer.zero_grad()
        z = self.encode()
        link_logits = self.decode(z, self.data.train_pos_edge_index, neg_edge_index)
        link_labels = self.get_link_labels(self.data.train_pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        self.optimizer.step()
        return loss


    @torch.no_grad()
    def test(self):
        self.eval()
        perfs = []
        for prefix in ["val", "test"]:
            pos_edge_index = self.data[f'{prefix}_pos_edge_index']
            neg_edge_index = self.data[f'{prefix}_neg_edge_index']
            z = self.encode()
            link_logits = self.decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
        return perfs


def main():
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device='cpu')
    model = model.float()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    best_val_perf = test_perf = 0
    model.modelin(optimizer)
    for epoch in range(1, 30):
        train_loss = model.trains()
        val_perf, tmp_test_perf = model.test()
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_loss, best_val_perf, test_perf))

    torch.save(model,"gcnmodel.model")

#main()