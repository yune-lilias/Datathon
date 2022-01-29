# Link & Recommendation, Rice Datathon 2022
---

![](./imgs/logo.jpg)

By [Tianjian Sun](https://github.com/TianjianSun), [Yuhan Yang](https://github.com/yune-lilias), [Haijiao Lu](https://github.com/LHJ98) and [Yun Sun](https://github.com/SophieSUN88).

---

## Project Description

This repository is for the project of Rice Datathon 2022, and this track is offered by Bill.com. These days, graph is widely used to represent data with much inter-relation. But most of time it's impossible to get a graph showing all the edges which exist, thus it's necessary to build a model learn from a part of the graph while most edges are missing, and let the model predicts existence of potential edges. This technique is a nature choice in making recommendation systems.

---

## What it does
Given part of a undirected graph with node & edge information, on one hand our project goal is to do a prediction on whether an edge (link) exist between two nodes, even if the edge information is not in our training set. On the other hand, the program can also find k node most likely to have connection with given node, like a recommendation system.

## How we built it
(1)As node features are various number of different "words" (index), to make it be a better input for our model, first our project need to vectorize the features. To make it clear, we map each the nodes into an 200-dimension vector. This can be easily accomplished by Doc2Vec model, which will put two node closer if their features are highly similar.
(2)In our project, we also use negative sampling method to give weight to edges "not-exist", otherwise our model will easily to make the prediction that all the pair of node is connected by an edge. Besides, we split our train data into three set: train set, validation set and test set to keep track of how well our model perform.
(3)Normal cnn/rnn works well if we just train on node features, but it fail to make use of edges information. For this reason we pick graph neural network by (py)torch-geometry in our project. Our neural network can be divided into two parts: one "encoder" to embed nodes, and one "decoder" to calculate "score" between each pair of nodes. In encoder, we use two different GNN network: one graph attention network and one graph convolution network, connected by a tanh activation layer.
(4)After reading testing node pair, our program will search the corresponding scores from score matrix, and let positive score to be "likely to have a edge" while negative score means the opposite.
(5)To make it easier to use, we also build a gui with edge prediction and node recommendation (print k nodes with highest score) based on user input node numbers and integer k, and some of necessary data is stored locally upon the first used.

## Challenges we ran into
(1)It's hard for us to build a gnn, which all of us have nearly no experience and knowledge about it.
(2)Running time is too high since model and score matrix is so large, make it slower to debug. Even with smaller dataset will takes quite a long time for pytorch to set up in pycharm or vscode. 
(3)After we choose to store some variable locally, the file size is also quite large. Which make it diffult to share with teammate because they exceed github file size limitation.
(4)Hard to find a way add regularization part to avoid overfitting
(5)Not enough time to read essays and compared with different gnn models in pytorch

## Accomplishments that we're proud of
(1)Our project can get high accuracy in validation and test(split from train) data, and we successfully finish the functionality finding k nodes have highest score with user input node number. 
(2)We build a nice gui for our project
(3)After the first search in gui, following search takes much less time (in 10s) 

## What's next for L&R
(1)Try to deploy the project on cloud workspace, which will save the time setting up gui and load local variables
(2)Make GUI more user-friendly
(3)Train multiply model, use mean of all models prediction results to be our output rather than single prediction.

## Built With
numpy & matplotlib: based vector calculation and plot function
Qtdesigner: build GUI
gensim: build doc2vec model to vectorize nodes
pandas: load csv to our program
pytorch-geometry: gnn related functions
