#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 3 \
    --hidden-dim 500 \
    --hidden-dim2 128 \
    --epochs 30 \
    --weight-decay 0 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.01 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
