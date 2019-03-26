from typing import Collection, Union, Tuple
from dataclasses import dataclass
from itertools import zip_longest

import torch
from torch import Tensor
from dependencies.text.torchtext.data.iterator import BucketIterator
from dependencies.text.torchtext.data.dataset import TabularDataset


@dataclass
class Batch:
    """Data Class representing a batch of data yield from the iterator.

    Attributes:
        src (Tensor): source tensor of shape `(batch_size, src_seq_len)`.
        src_mask (Tensor): mask over the source tensor of shape `(batch_size, src_seq_len)`.
        tgt_in (Tensor): target in tensor with <sos> token prepend of shape `(batch_size, tgt_seq_len + 1)`.
        tgt_out (Tensor): target out tensor with <eos> token append of shape `(batch_size, tgt_seq_len + 1)`.
        tgt_mask (Tensor): mask over the target tensors of shape `(batch_size, tgt_seq_len + 1)`.
        insert (Tensor): tensor of inserted words of shape `(batch_size, insert_seq_len)`.
        insert_mask (Tensor): mask over tensor of insert of shape `(batch_size, insert_seq_len)`.
        delete (Tensor): tensor of deleted words of shape `(batch_size, delete_seq_len)`.
        delete_mask (Tensor): mask over tensor of deleted words of shape `(batch_size, delete_seq_len)`.

    """
    src: Tensor
    src_mask: Tensor
    tgt_in: Tensor
    tgt_out: Tensor
    tgt_mask: Tensor
    insert: Tensor
    insert_mask: Tensor
    delete: Tensor
    delete_mask: Tensor


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
            Batch: a custom batch of data for the 'edit-transformer' forward function.

        """
        for batch in self.iterator:
            src = batch.src[0]
            src_mask = (src != self.pad_index)

            batch_size = batch.tgt[0].shape[0]
            tgt_in = torch.cat(
                [torch.tensor([[self.sos_index]] * batch_size, device=self.iterator.device), batch.tgt[0]], dim=-1)
            tgt_out = torch.cat(
                [batch.tgt[0], torch.tensor([[self.pad_index]] * batch_size, device=self.iterator.device)], dim=-1)
            tgt_out[range(batch_size), batch.tgt[1]] = self.eos_index
            tgt_mask = (tgt_in != self.pad_index)

            insert = batch.insert[0]
            insert_mask = (insert != self.pad_index)
            delete = batch.delete[0]
            delete_mask = (delete != self.pad_index)

            yield Batch(src, src_mask, tgt_in, tgt_out, tgt_mask, insert, insert_mask, delete, delete_mask)

    def __len__(self):
        """Return the same length as the original iterator."""
        return len(self.iterator)


def create_iterators(datasets: Collection[TabularDataset], batch_sizes: Union[int, Collection[int]],
                     repeats: Union[bool, Collection[bool]], device: torch.device, pad_index: int, sos_index: int,
                     eos_index: int) -> Tuple[IteratorWrapper, ...]:
    """ Create multiple iterators for training and evaluating at the same time.

    Args:
        datasets (Collection[TabularDataset]): a collection of datasets that will be used to create the iterators.
        batch_sizes (Union[int, Collection[int]]): either a single batch_size that will be used for all or each
            individual batch_size in a Collection.
        repeats (Union[bool, Collection[bool]]): boolean corresponding to the repeat parameter of each iterator.
        device (torch.device): the torch device on which to build the iterators upon.
        pad_index (int): the pad index used to generate the batches.
        sos_index (int): the sos index used to generate the batches.
        eos_index (int): the eos index used to generate the batches.

    Returns:
        Tuple[IteratorWrapper, ...]: a tuple of the same size of the dataset with each iterators.

    """
    num_iterator = len(datasets)
    batch_sizes = [batch_sizes] * num_iterator if isinstance(batch_sizes, int) else batch_sizes
    repeats = [repeats] * num_iterator if isinstance(repeats, bool) else repeats

    assert len(batch_sizes) == num_iterator
    assert len(repeats) == num_iterator

    iterators = []
    for dataset, batch_size, repeat in zip(datasets, batch_sizes, repeats):
        bucket_iterator = BucketIterator(dataset, batch_size=batch_size, repeat=repeat, device=device)
        iterators.append(IteratorWrapper(bucket_iterator, pad_index, sos_index, eos_index))

    return tuple(iterators)
