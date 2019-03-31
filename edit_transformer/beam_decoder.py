from typing import Tuple, Union, Iterable, List, Optional, Any
from queue import PriorityQueue
from dataclasses import dataclass, field
import math

import torch
from torch import LongTensor
import torch.nn.functional as F

from edit_transformer.model import EditTransformer
from edit_transformer.iterator import Batch


@dataclass
class Reference:
    """Store the references corresponding to a beam serch node.

    Attributes:
        src_sequence (LongTensor): the source sequence from which is derived a node of shape `(src_seq_len)`
        tgt_out_sequence (LongTensor): the out target sequence reference to a given node of shape `(tgt_seq_len)`
        insert_sequence (LongTensor): the filtered insert sequence from src to target of shape `(in_seq_len)`
        delete_sequence (LongTensor): the filtered delete sequence from src to target of shape `(in_seq_len)`

    """
    src_sequence: LongTensor
    tgt_out_sequence: LongTensor
    insert_sequence: LongTensor
    delete_sequence: LongTensor


@dataclass
class BeamSearchNode:
    """Store a single node from the beam search.

    Attributes:
        sequence (LongTensor): the sequence of indices corresponding to the node Tensor of shape `(seq_len)`.
        proba (float): the probability corresponding to this sequence.
        alpha (float): regularization parameter (higher alpha favors longer sentences).
        index (int): reference of the node (absolute position in the batch starting from 0 to batch_size).

    """
    sequence: LongTensor
    proba: float
    index: int
    alpha: float = 0.6

    def queue_score(self) -> float:
        """Return a priority queue score for the node (lowest are first).

        Returns:
            float: score that will be used by the PriorityQueue.

        """
        lp = ((5 + len(self.sequence)) ** self.alpha) / ((5 + 1) ** self.alpha)  # WU 2016
        return - math.log(self.proba) / lp


@dataclass(order=True)
class PrioritizedItem:
    """Priority item data class that implement sorting method using priority field only."""
    priority: float
    item: BeamSearchNode = field(compare=False)


class BeamSearchQueue(PriorityQueue):
    """An extend PriorityQueue to handle the beam search of a single batch sample.

    Attributes:
        eos_index (int): end of sentence index in use in the current model.
        beam_width (int): the size of the beam (number of results to wait for and return).
        max_len (int): the maximum length of a sequence.
        q_limit (int): the maximum number of nodes in the queue.
        finished_nodes (List[BeamSearchNode]): a list of beam search nodes completed.
        finished (bool): weather this queue is finished or not.

    """
    def __init__(self, eos_index: int, beam_width: int, max_len: int, q_limit: int = 500) -> None:
        """ Initialize a BeamSearchQueue.

        Args:
            eos_index (int): end of sentence index in use in the current model.
            beam_width (int): the size of the beam (number of results to wait for and return).
            max_len (int): the maximum length of a sequence.
            q_limit (int): the maximum number of nodes in the queue.

        """
        super().__init__()
        self.eos_index = eos_index
        self.beam_width = beam_width
        self.max_len = max_len
        self.q_limit = q_limit
        self.finished_nodes = []
        self.finished = False

    def put_nowait(self, item: Tuple[int, float, LongTensor]) -> None:
        """Create a new node and put it in the Queue.

        Args:
            item (Tuple[int, float, LongTensor]): tuple of index in the batch, probability, LongTensor of shape
                `(seq_len)`.

        """
        if not self.finished:
            if self.qsize() > self.q_limit:
                while len(self.finished_nodes) != self.beam_width:
                    self.finished_nodes.append(self.get_nowait().item)
                self.finished = True
            else:
                node = BeamSearchNode(sequence=item[2], proba=item[1], index=item[0])
                if node.sequence[-1].item() == self.eos_index or node.sequence.shape[0] == self.max_len:
                    self.finished_nodes.append(node)
                    if len(self.finished_nodes) == self.beam_width:
                        self.finished = True
                else:
                    super().put_nowait(PrioritizedItem(node.queue_score(), node))

    def get_nowait(self) -> Union[PrioritizedItem, None]:
        """Get the last best node if the Queue is not finished.

        Returns
            Union[Tuple[float, BeamSearchNode], None]: Either the score and its node or None if the Queue is finished.

        """
        if not self.finished:
            return super().get_nowait()
        else:
            return None


class BeamSearchQueueBatch(list):
    """A container of all the BeamSearchQueue of a batch.

    Attributes:
        pad_index (int): the index used for padding in the sequences.
        batch_size (int): the batch size wished for evaluation.

    """
    def __init__(self, pad_index: int, batch_size: int, queue_batch: Optional[Iterable[BeamSearchQueue]] = None):
        """ Initialize the container.

        Args:
            pad_index (int): the index used for padding in the sequences.
            queue_batch (Optional[Iterable[BeamSearchQueue]]): an optional iterable of beam search queue to initialize
                the container with.
            batch_size (int): the batch size wished for evaluation.

        """
        if queue_batch is not None:
            super().__init__(queue_batch)
        else:
            super().__init__()
        self.pad_index = pad_index
        self.batch_size = batch_size

    def all_queue_empty(self) -> bool:
        """Check if all contained queues are empty."""
        for queue in self:
            if not queue.empty():
                return False
        return True

    def all_queue_finished(self) -> bool:
        """Check if all queue are finished."""
        for queue in self:
            if not queue.finished:
                return False
        return True

    def get_next_batch_tensor(self) -> Union[Tuple[List[BeamSearchNode], LongTensor], Tuple[None, None]]:
        """ Get next batch tensor to pass through the model.

        Returns:
            Union[Tuple[List[BeamSearchNode], LongTensor], Tuple[None, None]]: a tuple composed of a list of current
                nodes being evaluated and the next padded tensor on which to run the model of shape
                `(batch, max_seq_len)`. It returns a tuple of None if every queue are finished.

        """
        if self.all_queue_finished():
            return None, None

        max_len = 0
        current_nodes: List[BeamSearchNode] = []

        while len(current_nodes) < self.batch_size and not self.all_queue_empty():
            for queue in self:
                if not queue.finished and not queue.empty():
                    node = queue.get_nowait().item
                    if node.sequence.shape[0] > max_len:
                        max_len = node.sequence.shape[0]
                    current_nodes.append(node)

        seq_to_stack = []
        for node in current_nodes:
            if node is not None:
                l = node.sequence.shape[0]
                seq_to_stack.append(
                    torch.cat((node.sequence, torch.tensor([self.pad_index] * (max_len - l), dtype=torch.long))))
            else:
                seq_to_stack.append(torch.tensor([self.pad_index] * max_len))

        return current_nodes, torch.stack(seq_to_stack)

    def get_final_nodes(self) -> List[List[BeamSearchNode]]:
        """Get the beam_width final nodes for each queue stored.

        Notes:
            it also make sure that all queue are actually in a `finished` state.

        Returns:
            List[List[BeamSearchNode]]: a list of list of BeamSearchNode of shape `(num_samples, beam_width)`.

        """
        for queue in self:
            assert queue.finished

        return [queue.finished_nodes for queue in self]


def beam_search(model: EditTransformer, batch: Batch, eos_index: int, pad_index: int, batch_size: int = 128,
                topk: int = 5, beam_width: int = 5, max_len: int = 50, draw_samples: bool = True,
                draw_p: bool = False) -> Tuple[List[List[BeamSearchNode]], List[Reference]]:
    """ Perform a beam_search on the provided data using the provided model.

    Args:
        model (EditTransformer): the model to use to perform the beam_search.
        batch (Batch): one `Batch` of inputs to the model.
        eos_index (int): index used to mark an end of sentence.
        pad_index (int): index used to pad the sequences too short.
        batch_size (int): the desired batch size to achieve when performing the beam search.
        topk (int): the number of best results to keep at each time-step.
        beam_width (int): the number of results to search for and output per sample.
        max_len (int): the maximum length of a sequence.
        draw_samples (bool): Weather to draw samples VAE style or not (keep True for training).
        draw_p (bool): Edit vector drawn from random prior distribution (keep False for training).

    Returns:
        Tuple[List[BeamSearchNode], List[Reference]]: a list of of list BeamSearchNode of len `(num_sample, beam_width)`
            and a list of corresponding reference data.

    """

    queue_batch = BeamSearchQueueBatch(pad_index, batch_size)
    tgt_in = batch.tgt_in[:, 0].unsqueeze(-1)
    tgt_mask = tgt_in != pad_index
    logits = model(batch.src, batch.src_mask, tgt_in, tgt_mask, batch.insert, batch.insert_mask, batch.delete,
                   batch.delete_mask, draw_samples, draw_p)
    proba = F.softmax(logits, dim=-1)
    topk_indices = torch.topk(proba, topk)[1][:, -1, :]  # shape `(batch, topk)`
    topk_proba = torch.topk(proba, topk)[0][:, -1, :]  # shape `(batch, topk)`

    for index in range(topk_indices.shape[0]):
        queue = BeamSearchQueue(eos_index, beam_width, max_len)
        for k in range(topk):
            queue.put_nowait((index, topk_proba[index][k].item(), topk_indices[index][k].long().unsqueeze(-1).cpu()))
        queue_batch.append(queue)

    current_nodes, batch_tensor = queue_batch.get_next_batch_tensor()
    while current_nodes is not None:
        sub = [n.index for n in current_nodes]
        tgt_in = batch_tensor.to(batch.src.device)
        tgt_mask = tgt_in != pad_index

        logits = model(batch.src[sub], batch.src_mask[sub], tgt_in, tgt_mask, batch.insert[sub], batch.insert_mask[sub],
                       batch.delete[sub], batch.delete_mask[sub], draw_samples, draw_p)
        logits[:, :, pad_index] = - float('inf')  # -> leads to a true 0 probability in the softmax for the padding.
        proba = F.softmax(logits, dim=-1)

        seq_indices = []
        for node in current_nodes:
            seq_indices.append(node.sequence.shape[0] - 1)

        topk_indices = torch.topk(proba, topk)[1][range(len(seq_indices)), seq_indices, :]  # shape `(batch, topk)`
        topk_proba = torch.topk(proba, topk)[0][range(len(seq_indices)), seq_indices, :]  # shape `(batch, topk)`

        for index in range(topk_indices.shape[0]):
            n = current_nodes[index]
            for k in range(topk):
                queue_batch[n.index].put_nowait((n.index, n.proba * topk_proba[index][k].item(), torch.cat(
                    (n.sequence, topk_indices[index][k].long().unsqueeze(-1).cpu()))))

        current_nodes, batch_tensor = queue_batch.get_next_batch_tensor()

    references = []
    for i in range(batch.tgt_out.shape[0]):
        tgt_out = batch.tgt_out[i]
        tgt_out = tgt_out[tgt_out != pad_index].cpu()
        src = batch.src[i]
        src = src[src != pad_index].cpu()
        insert = batch.insert[i]
        insert = insert[insert != pad_index].cpu()
        delete = batch.delete[i]
        delete = delete[delete != pad_index].cpu()
        references.append(Reference(src_sequence=src, tgt_out_sequence=tgt_out, insert_sequence=insert,
                                    delete_sequence=delete))

    return queue_batch.get_final_nodes(), references
