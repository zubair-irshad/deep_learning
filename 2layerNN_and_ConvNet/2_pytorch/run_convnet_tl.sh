#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train_transfer_learning.py \
    --epochs 20 \
    --weight_decay 1e-4 \
    --momentum 0.9 \
    --batch-size 100 \
    --lr 0.001 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
