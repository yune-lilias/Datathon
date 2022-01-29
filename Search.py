from gcn import Net
import torch
import torch_geometric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def interact(page, top_k):
    model = Net().to(device='cpu')
    model = torch.load("gcnmodel.model")
    model.eval()
    z = model.encode()
    if len(top_k) == 0:
        # test_data = pd.read_csv("datasets/Test Dataset/test_edges.csv")
        test_data = page
        test_data = test_data.values.tolist()
        test_data_nodes_score = model.edgepred(z, test_data)

        test_data_score = [float(test_data_nodes_score[i][2]) for i in range(len(test_data))]
        # print(test_data_score[:5])

        test_prediction = [1 if score > 0 else 0 for score in test_data_score]
        # print(test_prediction[:5])
        return test_prediction
    else:
        recommend_node, node_score = model.pred_one_edge(z, page, top_k)
        # print(recommend_node)

        # print(node_score)
        return recommend_node

def rEdge(Page):
    test_data_nodes_score = interact()
    return test_data_nodes_score


def rOne(z, Page1, Top_k):
    recommend_node, node_score = search.model.pred_one_edge(z, Page1, Top_k)
    return recommend_node, node_score



