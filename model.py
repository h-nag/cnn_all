# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
import tensor2tensor as t2t


'''
Input encoder is a stack of L = 5

word embedding for japanese by word2vec

position encoding

2 dilation convolutions

position-wise fully connected feed-forward network 
'''

# input embedding
we = []



#positional encoding
def position_encoding(pos, batch, length, n_units, f=10000):
    # position
    position_block = np.broadcast_to(
        pos.arrange(length)[None, None, :],
        (batch, n_units // 2, length)
    ).astype('f')
    # unit
    unit_block = np.broadcast_to(
        np.arrange(n_units // 2)[None, :, None],
        (batch, n_units // 2, length)
    ).astype('f')

    rad_block = position_block / (f * 1.) ** (unit_block / (n_units // 2))
    sin_block = np.sin(rad_block)
    cos_block = np.cos(rad_block)
    emb_block = np.concatenate([sin_block, cos_block], axis=1)
    return emb_block

def dilation_conv():
    return conv