"""Copy Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
from torch.distributions.uniform import Uniform
from torch.distributions.binomial import Binomial
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
        prob = 0.5 * torch.ones([seq_len,batch_size, seq_width], dtype=torch.float64)
        seq = Binomial(1, prob).sample()
        # Extra input channel for providing priority value
        inp = torch.zeros([seq_len, batch_size, seq_width + 1])
        inp[:seq_len, : , :seq_width] = seq

        # torch's Uniform function draws samples from the half-open interval
        # [low, high) but in the paper the priorities are drawn from [-1,1].
        # This minor difference is being ignored here as supposedly it doesn't
        # affects the task.
        priority = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        sorted = torch.zeros([seq_len, batch_size, seq_width + 1])
        for batch in range(batch_size):
            for i in range(seq_len):
                inp[i, batch, seq_width] = priority.sample()
            sorted[:,batch,:] = inp[torch.argsort(inp[:,batch,-1], 0, descending=True),batch,:]

        outp = sorted[:seq_len,:, :seq_width]

        yield batch_num+1, inp.float(), outp.float()


@attrs
class PrioritySortTaskParams(object):
    name = attrib(default="priority_sort")
    controller_size = attrib(default=200, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=5, convert=int)
    sequence_width = attrib(default=8, convert=int)
    sequence_len = attrib(default=20,convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=3e-5, convert=float)
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
class PrioritySortTaskModelTraining(object):
    params = attrib(default=Factory(PrioritySortTaskParams))
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
