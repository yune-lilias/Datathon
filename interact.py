from gcn import Net
import torch
import torch_geometric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def interact():
    model = Net().to(device='cpu')
    model = torch.load("gcnmodel.model")
    model.eval()
    z = model.encode()
    test_data = pd.read_csv("datasets/Test Dataset/test_edges.csv")
    test_data = test_data.values.tolist()
    test_data_nodes_score = model.edgepred(z, test_data)

    test_data_score = [float(test_data_nodes_score[i][2]) for i in range(len(test_data))]
    print(test_data_score[:5])

    test_prediction = [1 if score > 0 else 0 for score in test_data_score]
    print(test_prediction[:5])

    recommend_node, node_score = model.pred_one_edge(z, 1, top_k=4)
    print(recommend_node)

    print(node_score)

    with open('trainrecord.npy','rb') as f:
        xrecord = np.load(f)
    with open('validrecord.npy','rb') as f:
        vrecord = np.load(f)
    with open('testrecord.npy','rb') as f:
        trecord = np.load(f)

    fig ,ax = plt.subplots(3,1)
    ax[0].plot(xrecord)
    ax[0].title.set_text("train loss")
    ax[1].plot(vrecord)
    ax[1].title.set_text("validate accuracy")
    ax[2].plot(trecord)
    ax[2].title.set_text("test accuracy")

    plt.show()


interact()