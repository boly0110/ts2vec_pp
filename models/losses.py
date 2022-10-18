import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = paddle.to_tensor(0.)
    d = 0
    while z1.shape[1] > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(perm=[0,2,1]), kernel_size=2).transpose(perm=[0,2,1])
        z2 = F.max_pool1d(z2.transpose(perm=[0,2,1]), kernel_size=2).transpose(perm=[0,2,1])
    if z1.shape[1] == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.shape[0], z1.shape[1]
    if B == 1:
        return paddle.to_tensor(0.)
    z = paddle.concat([z1, z2], axis=0)  # 2B x T x C
    z = z.transpose(perm=[1,0,2])  # T x 2B x C
    sim = paddle.matmul(z, z.transpose(perm=[0,2,1])) # T x 2B x 2B
    logits = paddle.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += paddle.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, axis=-1)
    
    loss = (logits[:,0:B, B-1:(2*B-1)].mean() + logits[:, B:2*B, 0:B].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.shape[0], z1.shape[1]
    if T == 1:
        return paddle.to_tensor(0.)
    z = paddle.concat([z1, z2], axis=1)  # B x 2T x C
    sim = paddle.matmul(z, z.transpose(perm=[0,2,1]))  # B x 2T x 2T
    logits = paddle.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += paddle.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, axis=-1)
    
    loss = (logits[:, 0:T, T-1:(2*T-1)].mean() + logits[:, T:2*T, 0:T].mean()) / 2
    return loss
