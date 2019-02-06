# Hongkong-Project

## Features

### alpha

Constant that multiplies the regularization term.

### l1\_ratio

The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.

### max\_iter

The maximum number of passes over the training data (aka epochs).

### make\_classification

This initially creates clusters of points normally distributed (std=1) about vertices of an n_informative-dimensional hypercube with sides of length 2*class_sep and assigns an equal number of clusters to each class. It introduces interdependence between these features and adds various types of further noise to the data.

### SGDClassifier

This estimator implements **regularized linear models** with **stochastic gradient descent** (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).