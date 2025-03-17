# Implementation of a Feed Forward Neural Network using Fashion MNIST Dataset

The GitHub Repository can be find here- https://github.com/Kahkashan2708/Deep_Learning-Assignment-1
Wandb Report- https://wandb.ai/ma23c014-indian-institute-of-technology-madras/fashion-mnist-sweep/reports/DA6401-Assignment-1-Implementing-a-Feedforward-Neural-Network-for-Fashion-MNIST-Classification--VmlldzoxMTgyMjIzOQ

The problem statement involves building and training a 'plain vanilla' Feed Forward Neural Network from scratch using primarily Numpy package in Python.

The code base now has the following features:

1. Forward and backward propagation are hard coded using Matrix operations. The weights and biases are stored separately as dictionaries to go hand in hand with the notation used in class.
2. A neural network class to instantiate the neural network object for specified set of hyperparameters, namely the number of layers, hidden neurons, activation function, optimizer, weight decay,etc.
3. The optimisers, activations and their gradients are passed through dictionaries configured as attributed within the FeedForwardNeuralNetwork class.
4. All the code from are there separately in train.py file.
5. Colab notebook and jupyter notebook containing the entire code to train and validate the model from scratch

# DATASET

Fashion MNIST data set has been used here in this assignment instead of the traditional MNIST hand written digits dataset. Train - 60000 Test - 20000 Validation - 6000

For the hyper parameter optimisation stage, 10% of the randomly shuffled training data set (around 6000 images and corresponding labels) are kept aside for validation for each hyperparameter configuration while the model is trained on the remaining 54000 images from the randomly shuffled training data set.

Once the best configuration is identified with the help of wandb wither using Random search or Bayesian optimisation, the full training dataset is used to train the best model configuration and the test accuracy is calculated. The resulting confusion matrix is plotted therafter.

# Code Structure
train.py - contains all the code in python file

Class_samples&FFNN.ipynb - This file contains fashion-mnist-visualization nd code of feedforward neural network

confusion_matrix.ipynb - code of confusion matrix and visualization.

Digit-mnist.ipynb - contains code of the implementation of ffnt and hyperparameter tuning of digit-mnist-dataset.

# Training, Validation and Hyperparameter optimisation

Run Fashion-mnist-sweep for hyperparameter Tuning.

sweep_config = {
  "name": "Random Sweep", #(or) Bayesian Sweep (or) Grid search
  "method": "random", #(or) bayes (or) grid
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  },
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER"]
        },

        "num_layers": {
            "values": [2, 3, 4]
        },
        
        
        "num_hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation": {
            "values": ['RELU', 'SIGMOID', 'TANH']
        },
        
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        
        
        "weight_decay": {
            "values": [0, 0.0005,0.5]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        }
        
        
    }
}
One can choose to select / modify/omit any of the hyperparameters above in the config dictionary.

# Results:
For the plain vanilla feed forward neural network implemented, the maximum test accuracy reported was 88.08% on the Fashion MNIST dataset and ~90.2% on the MNIST hand written datasets. One of the model configuration chosen to be the best is as follows:

Number of Hidden Layers - 3
Number of Hidden Neurons - 128
L2 Regularisation - No
Activation - Sigmoid
Initialisation - Xavier
Optimiser - NADAM
Learning Rate - 0.001
Batch size - 32
