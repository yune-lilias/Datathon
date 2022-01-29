import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import max_pool
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F

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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.nl = BatchNorm(128)
        self.conv2 = GCNConv(128, 64)

    def encode(self):
        x = self.conv1(data.x, data.train_pos_edge_index)
        x = x.relu()
        return self.conv2(x, data.train_pos_edge_index)

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
        rank_list = np.zeros(data.num_nodes)
        for k in range(data.num_nodes):
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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model, data = Net().to(device), data.to(device)
model = model.float()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
        force_undirected=True,
    )
    optimizer.zero_grad()
    z = model.encode()
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode()
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return perfs

best_val_perf = test_perf = 0
for epoch in range(1, 30):
    train_loss = train()
    val_perf, tmp_test_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_perf, test_perf))
z = model.encode()

test_data = pd.read_csv("datasets/Test Dataset/test_edges.csv")
test_data = test_data.values.tolist()
test_data_nodes_score = model.edgepred(z, test_data)

test_data_score = [float(test_data_nodes_score[i][2]) for i in range(len(test_data))]
print(test_data_score[:5])

test_prediction = [1 if score >0 else 0 for score in test_data_score]
print(test_prediction[:5])

recommend_node, node_score = model.pred_one_edge(z, 1, top_k = 4)
print(recommend_node)

print(node_score)