# ERA1_Session8
## Assignment
    1. Use CIFAR10 dataset 
    2. Make this network: 
        1. C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10 (caps-C means 3x3 convolution, lower-c means 1x1 convolution and P means Maxpool layer)
        2. Keep the parameter count less than 50000 
        3. Max Epochs is 20 
    3. Create 3 versions of the above code (in each case achieve above 70% accuracy): 
        1. Network with Batch Normalization 
        2. Network with Group Normalization 
        3. Network with Layer Normalization 

## 1. Network with Batch Normalization - Notebook [s8_bn_1](https://github.com/sdev2030/ERA1_Session8/s8_bn_1.ipynb)
In this first network we will use batch normalization layer after each convolution layer except in the final **c10** convolution layer. 
Following are the model parameter, train and test accuracies achieved in training the model for 20 epochs.
- Model Parameters - 34320
- Train Accuracy - 79.68%
- Test Accuracy - 79.00%

Graphs from training showing loss and accuracy for train and test datasets
![Batch Norm Graphs](https://github.com/sdev2030/ERA1_Session8/blob/main/images/BN_training_graphs.png)

Ten misclassified images from the batchnorm trained model.
![Batch Norm misclassied images](https://github.com/sdev2030/ERA1_Session8/blob/main/images/BN_wrong_classified.png)

## 2. Network with Group Normalization - Notebook [s8_gn_2](https://github.com/sdev2030/ERA1_Session8/s8_gn_2.ipynb)
In this network we will use group normalization layer with group size of 4 after each convolution layer except in the final **c10** convolution layer. 
Following are the model parameter, train and test accuracies achieved in training the model for 20 epochs.
- Model Parameters - 34320
- Train Accuracy - 78.84%
- Test Accuracy - 77.00%

Graphs from training showing loss and accuracy for train and test datasets
![Group Norm Graphs](https://github.com/sdev2030/ERA1_Session8/blob/main/images/GN_training_graphs.png)

Ten misclassified images from the batchnorm trained model.
![Group Norm misclassied images](https://github.com/sdev2030/ERA1_Session8/blob/main/images/GN_wrong_classified.png)

## 3. Network with Layer Normalization - Notebook [s8_ln_3](https://github.com/sdev2030/ERA1_Session8/s8_ln_3.ipynb)
In this network we will use layer normalization layer after each convolution layer except in the final **c10** convolution layer. 
Following are the model parameter, train and test accuracies achieved in training the model for 20 epochs.
- Model Parameters - 34320
- Train Accuracy - 78.95%
- Test Accuracy - 77.31%

Graphs from training showing loss and accuracy for train and test datasets
![Layer Norm Graphs](https://github.com/sdev2030/ERA1_Session8/blob/main/images/LN_training_graphs.png)

Ten misclassified images from the batchnorm trained model.
![Layer Norm misclassied images](https://github.com/sdev2030/ERA1_Session8/blob/main/images/LN_wrong_classified.png)

In all three networks we could achive higher than 70% test accuracy.  