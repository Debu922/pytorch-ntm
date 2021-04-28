"""Associate recall task"""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM
from torch.distributions.binomial import Binomial


# Generator of randomized test sequences
def dataloader(num_batches,
               batch_size,
               seq_width,
               seq_len,
               min_items,
               max_items):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(num_batches):

        n_items = random.randint(min_items, max_items)    
        # print(n_items)
        binom_prob = 0.5*torch.ones([seq_len, batch_size, seq_width], dtype=torch.float64)    
        seq = Binomial(1, binom_prob)
        # seq = torch.from_numpy(seq)
        # print(seq)
        inp = torch.zeros([(seq_len+1)*(n_items+1)+1, batch_size, seq_width+2])

        for i in range(n_items):
            inp[(seq_len+1)*i, :, seq_width] = 1.0
            inp[(seq_len+1)*i+1:(seq_len+1)*(i+1), :, :seq_width] = seq.sample()

        query_item = torch.randint(0, n_items - 1, (1,), dtype=torch.long).item()

        query_seq = inp[(seq_len+1)*query_item+1:(seq_len+1)*(query_item+1), :, :seq_width]

        inp[(seq_len+1)*n_items, :, seq_width+1] = 1.0
        inp[(seq_len+1)*n_items+1:(seq_len+1)*(n_items+1),:, :seq_width] = query_seq
        inp[(seq_len+1)*(n_items+1), :, seq_width+1] = 1.0

        outp = torch.zeros([seq_len, batch_size, seq_width])

        if query_item != n_items-1:
            outp[:seq_len, :, :seq_width] = inp[(seq_len+1)*(query_item+1)+1:(seq_len+1)*(query_item+2), :, :seq_width]

        # A2ll batches have the same sequence length
        # seq_len = random.randint(min_len, max_len)
        # seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        # seq = torch.from_numpy(seq)

        # # The input includes an additional channel used for the delimiter
        # inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        # inp[:seq_len, :, :seq_width] = seq
        # inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
        # outp = seq.clone()

        # print(inp.shape)
        # print(outp.shape)

        yield batch_num+1, inp.float(), outp.float()


@attrs
class AssociateRecallParams(object):
    name = attrib(default="recall-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=10, convert=int)
    sequence_length = attrib(default=3, convert=int)
    minimum_items = attrib(default=2, convert=int)
    maximum_items = attrib(default=6, convert=int)
    # sequence_min_len = attrib(default=1,convert=int)
    # sequence_max_len = attrib(default=20, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)


#
# To create a network simply instantiate the `:class:CopyTaskModelTraining`,
# all the components will be wired with the default values.
# In case you'd like to change any of defaults, do the following:
#
# > params = CopyTaskParams(batch_size=4)
# > model = CopyTaskModelTraining(params=params)
#
# Then use `model.net`, `model.optimizer` and `model.criterion` to train the
# network. Call `model.train_batch` for training and `model.evaluate`
# for evaluating.
#
# You may skip this alltogether, and use `:class:CopyTaskNTM` directly.
#

@attrs
class RecallTaskModelTraining(object):
    params = attrib(default=Factory(AssociateRecallParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.params.sequence_width + 2, self.params.sequence_width,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_width, self.params.sequence_length, 
                          self.params.minimum_items, self.params.maximum_items)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
