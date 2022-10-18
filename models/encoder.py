import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
import math

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = paddle.full((B, T), True, dtype='bool')
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return paddle.to_tensor(np.random.binomial(1, p, size=(B, T)), dtype='bool')


class TSEncoder(nn.Layer):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super(TSEncoder, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-math.sqrt(1./input_dims), math.sqrt(1./input_dims))),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-math.sqrt(1./input_dims), math.sqrt(1./input_dims)))
        )
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.shape[0], x.shape[1])
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.shape[0], x.shape[1])
        elif mask == 'all_true':
            mask = paddle.full((x.shape[0], x.shape[1]), True, dtype='bool')
        elif mask == 'all_false':
            mask = paddle.full((x.shape[0], x.shape[1]), False, dtype='bool')
        elif mask == 'mask_last':
            mask = paddle.full((x.shape[0], x.shape[1]), True, dtype='bool')
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(perm=[0,2,1]) # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(perm=[0,2,1])  # B x T x Co
        return x
        