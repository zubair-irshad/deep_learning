#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 3 \
    --hidden-dim 512 \
    --hidden-dim2 128 \
    --epochs 20 \
    --weight_decay 1e-5 \
    --momentum 0.9 \
    --batch-size 128 \
    --lr 0.01 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
