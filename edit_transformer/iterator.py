from dataclasses import dataclass

import torch
from torch import Tensor
from torchtext.data.iterator import BucketIterator


@dataclass
class Batch:
    """Data Class representing a batch of data yield from the iterator.

    Attributes:
        src (Tensor):
        src_mask (Tensor):
        tgt_in (Tensor):
        tgt_out (Tensor):
        tgt_mask (Tensor):
        insert (Tensor):
        delete (Tensor):

    """
    src: Tensor
    src_mask: Tensor
    tgt_in: Tensor
    tgt_out: Tensor
    tgt_mask: Tensor
    insert: Tensor
    delete: Tensor


class IteratorWrapper:
    """Wrapper class around BucketIterator to yield custom batches."""
    def __init__(self, iterator: BucketIterator, pad_index: int, sos_index: int, eos_index: int):
        """Initialize the Iterator Wrapper with the original iterator.

        Args:
            iterator (BucketIterator): Iterator to wrap which yields batches with the following attributes:
                {
                    'src (Tuple[Tensor, Tensor])':
                    'tgt (Tuple[Tensor, Tensor])':
                    'insert (Tuple[Tensor, Tensor])':
                    'delete (Tuple[Tensor, Tensor])':
                }
            pad_index (int): index of the pad token used.
            sos_index (int): index of the start of sentence token used.
            eos_index (int): index of the end of sentence token used.

        """
        self.iterator = iterator
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index

    def __iter__(self):
        """Iterate over the original iterator and generates missing masks and add <eos> and <sos>.

        Yields:
            Batch: a custom batch of data for the 'edit-transformer' forward.

        """
        for batch in self.iterator:
            src = batch.src[0]
            src_mask = src != self.pad_index

            batch_size = batch.tgt[0].shape[0]
            tgt_in = torch.cat([torch.tensor([[self.sos_index]] * batch_size), batch.tgt[0]], dim=-1)
            tgt_out = torch.cat([batch.tgt[0], torch.tensor([[self.pad_index]] * batch_size)], dim=-1)
            tgt_out[range(batch_size), batch.tgt[1]] = self.eos_index
            tgt_mask = tgt_in != self.pad_index

            import ipdb; ipdb.set_trace()
            yield Batch(src, src_mask, tgt_in, tgt_out, tgt_mask, batch.insert[0], batch.delete[0])

    def __len__(self):
        """Return the same length as the original iterator."""
        return len(self.iterator)
