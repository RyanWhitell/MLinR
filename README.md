# An Introduction to Machine Learning in R

### K-Nearest Neighbor
#### Predicting Heart Disease in Patients
K-nearest neighbors is an extremely simple classification and regression algorithm that classifies or predicts a data point based on the majority vote of its nearest neighbors. The important characteristics of KNN are how many neighbors to consider (K) and the method used to calculate distance.
___
### Decision Trees and Random Forests
#### Predicting the Quality of Wine
Decision trees are perhaps the most intuitive classification models for humans to interpret. Decision tree algorithms build trees by splitting data based off of some sort of information measure recursively until a stopping point. The many nuances of improving performance are omitted from this discussion but can easily be found online. The important characteristic is that each node contains a subset of the data, a node is split in such a way as to minimize the amount of diversity of its children.  

Random forests are an ensemble method for decisions trees. Many trees are built using different methods, usually by taking subsets of the training data. The classification of new data is the majority vote of all the trees (classification) or the average value (regression). Random forests generalize data better than single decision trees.
___
### Naïve Bayes
#### Filtering SMS Spam
Naïve Bayes classifiers use Bayes rule to classify data using the probabilities gleaned from existing data or new data as it becomes available. Naïve Bayes is very good at text classification even with minimal data preprocessing or wrangling.
___
### Artificial Neural Networks
#### Predicting the Edibility of Mushrooms
Artificial neural networks borrow from biology. Like their namesake, artificial neural networks model the neural connections present in our own brains. Nodes in a network are connected to inputs and outputs. Whether or not the output is activated depends on the inputs, the weight on those inputs, and the activation function. Putting many nodes together results in a network that is able to represent any function. What is learned or trained in an ANN are the weights on each input.
___
### Support Vector Machines
#### Predicting the Edibility of Mushrooms
Support Vector Machines are not really “machines” but more a clever algorithm with a simple concept that has a complicated implementation. The simple concept is just a hyperplane that separates classes with the widest possible margin between the hyperplane and closest data points. New examples will be classified by which side of the hyperplane (decision boundary) they fall. The complicated, and clever, implementation of this algorithm involves finding this hyperplane using vectors and quadratic programming. Basically, only the points closest to the decision boundary will be considered (the support vectors). For data sets that are not linearly separable, the kernel trick is used.
___
### Evaluation Metrics
#### Classifying Abalone Age
Choosing the correct evaluation metrics are important when evaluating a model. Accuracy is a simple metric that is able to provide quick feedback for how good a model is, but for skewed data it can be misleading. For example, if the data contains 99990 examples with class ‘0’ and 10 examples of class ‘1,’ the model will be 99.99% accurate by always guessing ‘0.’ As another example, take a neural net with 88 outputs indicating keys on a piano, if the ANN always guesses that no note is played (all zeros), it is still around 98% accurate.

There are many different metrics to consider, the best one depends on the goal of the model. In the case of cancer diagnosis, it is much better to lean towards a positive diagnosis even when there is no cancer (false positive) than to have a negative diagnosis when there is a presence of cancer (false negative). For spam detection, it is better to let some spam through if it means never blocking ham messages. The model should be evaluated based on the most important metric.

This project uses the `caret` package to tune and train a Multi-Layer Perceptron (ANN) and a SVM for the purpose of investigating different evaluation metrics.
___
### K-Means and Hierarchical Cluster Analysis
#### Exploring Wholesale Data
K-means and hierarchical cluster analysis (HCA) are both unsupervised learning techniques that describe data with clusters.
___
### Reinforcement Learning
#### Solving the Tower of Hanoi and Playing Tic-Tac-Toe
Reinforcement learning is a more diverse area of machine learning than supervised and unsupervised learning. The simple idea behind reinforcement learning comes from behaviorist psychology. That is, in some state, an agent will take some action that it has learned will maximize its instant or future reward. There are many different types of reinforcement learning problems and many different types of algorithms to solve them. Using the `ReinforcementLearning` package in R, a policy can be learned from examples of state-action-reward transitions.
