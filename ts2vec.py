import paddle
import paddle.nn.functional as F
from paddle.io import Dataset, TensorDataset, DataLoader
from typing import Union, Callable, Optional, List
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, paddle_pad_nan
import math


class ts2vecDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class TS2Vec:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims: int,
        output_dims: int=320,
        hidden_dims: int=64,
        depth: int=10,
        lr: float=0.001,
        batch_size: int=16,
        max_train_length=None,
        temporal_unit: int=0,
        after_iter_callback: Union[Callable, None] = None,
        after_epoch_callback: Union[Callable, None] = None):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.net = paddle.incubate.ModelAverage(0.15, parameters=self._net.parameters())
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        train_dataset = ts2vecDataset(paddle.to_tensor(train_data, dtype='float32'))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = paddle.optimizer.AdamW(learning_rate=self.lr, parameters=self._net.parameters())
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader():
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                x = batch
                if self.max_train_length is not None and x.shape[1] > self.max_train_length:
                    window_offset = np.random.randint(x.shape[1] - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]                
                
                ts_l = x.shape[1]
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.shape[0])

                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )
                
                loss.backward()
                optimizer.step()
                self.net.step()
                #self.net.apply(need_restore=False)
                optimizer.clear_grad()
                self.net.clear_grad()
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self._net(x, mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(perm=[0,2,1]),
                kernel_size = out.shape[1],
            ).transpose(perm=[0,2,1])
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(perm=[0,2,1]),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(perm=[0,2,1])
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.shape[1]:
                t_out = F.max_pool1d(
                    out.transpose(perm=[0,2,1]),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(perm=[0,2,1])
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = paddle.concat(reprs, axis=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out
    
    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        dataset = ts2vecDataset(paddle.to_tensor(data, dtype='float32'))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        self._net.eval()
        self.net.apply(need_restore=False)
        with paddle.no_grad():
            output = []
            for batch in loader():
                x = batch
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = paddle_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    paddle.concat(calc_buffer, axis=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += paddle.split(out, out.shape[0]//n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                paddle.concat(calc_buffer, axis=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += paddle.split(out, out.shape[0]//n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = paddle.concat(reprs, axis=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(perm=[0,2,1]),
                            kernel_size = out.shape(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = paddle.concat(output, axis=0)
            
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        paddle.save(self._net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = paddle.load(fn)
        self._net.set_state_dict(state_dict)
    
