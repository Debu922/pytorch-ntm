"""Lexicographic Sort Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



# Generator of randomized test sequences
def dataloader(num_batches,
               batch_size,
               seq_width,
               seq_len):
    """Generator of random sequences for the Sort task.

    Creates random batches of "bits" sequences. Which needs to be 
    sorted by the NTM according to number of bits.

    All the sequences within each batch have the same length.
    The length is 'seq_len'

    :param num_batches: Total number of batches to generate.
    :param batch_size: Batch size.
    :param seq_width: The width of each item in the sequence.
    :param seq_len: The Legnth of each sequence in the batch.

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(num_batches):
        
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        inp = torch.zeros((2 * seq_len + 1, batch_size, seq_width+1))
        outp = torch.zeros((2 * seq_len + 1, batch_size, seq_width))
        sorted_seq = np.empty(shape = seq.shape)
        for index, inner in enumerate(seq):
            sorted_seq[index] = inner[np.lexsort(inner.T[::-1])]
        seq = torch.from_numpy(seq)
        sorted_seq = torch.from_numpy(sorted_seq)
        inp[:seq_len, :, :seq_width] = seq
        inp[:seq_len, :, -1] = 1
        outp[seq_len+1:2*seq_len+1, :, :seq_width] = sorted_seq
    
        yield batch_num+1, inp.float(), outp.float()


@attrs
class SortTaskParams(object):
    name = attrib(default="sort")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=8, convert=int)
    sequence_len = attrib(default=20,convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)


#
# To create a network simply instantiate the `:class:SortTaskModelTraining`,
# all the components will be wired with the default values.
# In case you'd like to change any of defaults, do the following:
#
# > params = SortTaskParams(batch_size=4)
# > model = SortTaskModelTraining(params=params)
#
# Then use `model.net`, `model.optimizer` and `model.criterion` to train the
# network. Call `model.train_batch` for training and `model.evaluate`
# for evaluating.
#
# You may skip this alltogether, and use `:class:SortTaskNTM` directly.
#

@attrs
class SortTaskModelTraining(object):
    params = attrib(default=Factory(SortTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.params.sequence_width + 1, self.params.sequence_width,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)

        # return net.to(device)
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_width,
                          self.params.sequence_len)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
