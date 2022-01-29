from gcn import Net
import torch
import torch_geometric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
model = Net().to(device='cpu')
model = torch.load("gcnmodel.model")
model.eval()
z = model.encode()

def interact(page, top_k = 0):

    if top_k == 0:
        # test_data = pd.read_csv("datasets/Test Dataset/test_edges.csv")
        test_data = page
        #test_data = test_data.values.tolist()
        test_data_nodes_score = model.edgepred(z, test_data , savep = True)

        test_data_score = [float(test_data_nodes_score[i][2]) for i in range(len(test_data))]
        # print(test_data_score[:5])

        test_prediction = [1 if score > 0 else 0 for score in test_data_score]
        # print(test_prediction[:5])
        return test_prediction
    else:
        recommend_node, node_score = model.pred_one_edge(z, page, top_k, savep = True)
        # print(recommend_node)

        # print(node_score)
        return recommend_node
