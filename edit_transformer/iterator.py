from __future__ import annotations
from typing import Collection, Union, Tuple, Any
from dataclasses import dataclass

import torch

from dependencies.typing import T_LongTensor
from dependencies.text.torchtext.data.iterator import BucketIterator
from dependencies.text.torchtext.data.dataset import TabularDataset
from dependencies.text.torchtext.data.batch import Batch as TextBatch


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
        batch_size (int): size of the batch.

    """
    src: T_LongTensor
    src_mask: T_LongTensor
    tgt_in: T_LongTensor
    tgt_out: T_LongTensor
    tgt_mask: T_LongTensor
    insert: T_LongTensor
    insert_mask: T_LongTensor
    delete: T_LongTensor
    delete_mask: T_LongTensor
    batch_size: int

    def __len__(self):
        return self.batch_size

    @classmethod
    def from_iterator_batch(cls, batch: TextBatch, pad_index: int, sos_index: int, eos_index: int) -> Batch:
        """Create a batch from an iterator batch.

        Args:
            batch (TextBatch): a batch from iterator to wrap into `Batch` with the following attributes:
                {
                    'src (Tuple[T_LongTensor, T_LongTensor])': tuple of tensors containing the source sentence
                        numericalized and padded of shape `(batch_size, seq_len)` and the length of each sentence of
                        shape `(batch_size)`.
                    'tgt (Tuple[T_LongTensor, T_LongTensor])': tuple of tensors containing the target sentence
                        numericalized and padded of shape `(batch_size, seq_len)` and the length of each sentence of
                        shape `(batch_size)`.
                    'insert (Tuple[T_LongTensor, T_LongTensor])': tuple of tensors containing the inserted word
                        numericalized and padded of shape `(batch_size, seq_len)` and the length of each sequence of
                        words of shape `(batch_size)`.
                    'delete (Tuple[T_LongTensor, T_LongTensor])': tuple of tensors containing the deleted word
                        numericalized and padded of shape `(batch_size, seq_len)` and the length of each sequence of
                        words of shape `(batch_size)`.
                }
            pad_index (int): index of the pad token used.
            sos_index (int): index of the start of sentence token used.
            eos_index (int): index of the end of sentence token used.

        Returns:
            Batch: a newly created batch input for the model.

        """
        batch_size = len(batch)
        src = torch.cat(
            [batch.src[0], torch.tensor([[pad_index]] * batch_size, device=batch.src[0].device)], dim=-1)
        src[range(batch_size), batch.src[1]] = eos_index
        src_mask = (src != pad_index)


        tgt_in = torch.cat(
            [torch.tensor([[sos_index]] * batch_size, device=batch.tgt[0].device), batch.tgt[0]], dim=-1)
        tgt_out = torch.cat(
            [batch.tgt[0], torch.tensor([[pad_index]] * batch_size, device=batch.tgt[0].device)], dim=-1)
        tgt_out[range(batch_size), batch.tgt[1]] = eos_index
        tgt_mask = (tgt_in != pad_index)

        insert = batch.insert[0]
        insert_mask = (insert != pad_index)
        delete = batch.delete[0]
        delete_mask = (delete != pad_index)

        return cls(src, src_mask, tgt_in, tgt_out, tgt_mask, insert, insert_mask, delete, delete_mask, batch_size)

    @classmethod
    def from_src_seq_only(cls, src: T_LongTensor, pad_index: int, sos_index: int, eos_index: int) -> Batch:
        """Create a batch from a source sequences only.

        Args:
            src (T_LongTensor): the source sequences on which the batch will be created of shape `(batch_size, seq_len)`
            pad_index (int): index of the pad token used.
            sos_index (int): index of the start of sentence token used.
            eos_index (int): index of the end of sentence token used.

        Returns:
            Batch: a newly created batch input for the model.

        """
        src_mask = (src != pad_index)

        batch_size = src.shape[0]
        tgt_in = torch.tensor([[sos_index]] * batch_size, device=src.device)
        tgt_out = torch.tensor([[eos_index]] * batch_size, device=src.device)
        tgt_mask = (tgt_in != pad_index)

        insert = torch.zeros(batch_size, 1, dtype=torch.long, device=src.device)
        insert_mask = (insert != pad_index)
        delete = torch.zeros(batch_size, 1, dtype=torch.long, device=src.device)
        delete_mask = (delete != pad_index)

        return cls(src, src_mask, tgt_in, tgt_out, tgt_mask, insert, insert_mask, delete, delete_mask, batch_size)


class IteratorWrapper:
    """Wrapper class around BucketIterator to yield custom batches."""
    def __init__(self, iterator: BucketIterator, pad_index: int, sos_index: int, eos_index: int) -> None:
        """Initialize the Iterator Wrapper with the original iterator.

        Args:
            iterator (BucketIterator): Iterator to wrap which yields batches object.
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
            yield Batch.from_iterator_batch(batch, self.pad_index, self.sos_index, self.eos_index)

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
