from typing import Tuple, List, Optional
from logging import Logger
from collections import Counter
import math

from torch import LongTensor
import torch.nn.functional as F
from dependencies.text.torchtext.vocab import Vocab
from tensorboardX import SummaryWriter

from edit_transformer.model import EditTransformer
from edit_transformer.iterator import IteratorWrapper
from edit_transformer.beam_decoder import beam_search, BeamSearchNode, Reference


def compute_loss(model: EditTransformer, iterator: IteratorWrapper, limit: int, pad_index: int) -> float:
    """ Compute the nll loss over `limit` batches of the provided iterator.

    Args:
        model (EditTransformer): the model used to compute the loss.
        iterator (IteratorWrapper): the iterator from which `Batch` is yield.
        limit (int): the limit over the number of iterator to evaluate.
        pad_index (int): the padding index to ignore when computing the loss.

    Returns:
        float: the average negative loglikelihood loss over the batches.

    """
    losses = []
    for batch in iterator:
        if iterator.iterator.iterations > limit:
            break
        logits = model(batch.src, batch.src_mask, batch.tgt_in, batch.tgt_mask, batch.insert, batch.insert_mask,
                       batch.delete, batch.delete_mask)
        loss = F.nll_loss(F.log_softmax(logits.transpose(1, 2), dim=1), batch.tgt_out, ignore_index=pad_index)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def tensor_to_sentence(text_seq: LongTensor, vocab: Vocab) -> str:
    """Transform a tensor (sequence of indices) into some readable text.

    Args:
        text_seq (LongTensor): tensor sequence of indices of shape `(seq_len)`.
        vocab (Vocab): vocabulary object used with the model that generated the text_seq.

    """
    return " ".join([vocab.itos[v.item()] for v in text_seq])


class ExamplesWriter:
    """Class to write the computed examples of generation.

    Attributes:
        path (str): path to write the examples generated.

    """
    def __init__(self, path: str) -> None:
        """Initialize the ExamplesWriter.

        Args:
            path(str): the path to the file where to save the examples.

        """
        self.path = path

    def write_samples(self, iteration: int, nodes_list: List[List[BeamSearchNode]],
                      references: List[Reference], vocab: Vocab) -> None:
        """ Append the provided samples to the file specified in the path attribute.

        Args:
            iteration (int): the number of iteration of training before obtaining those results.
            nodes_list (List[List[BeamSearchNode]]): a result of a beam search on the model evaluated.
            references (List[Reference]): the references corresponding to the nodes_list.
            vocab (Vocab): the vocab used in the tensor passed to the writer.

        """
        with open(self.path, mode="a", encoding="utf8") as f:
            f.write("=" * 20 + " ITER #{} ".format(iteration) + "=" * 20 + "\n")
            for nodes, reference in zip(nodes_list, references):
                f.write("SRC | {}\n".format(tensor_to_sentence(reference.src_sequence, vocab)))
                f.write("TGT | {}\n".format(tensor_to_sentence(reference.tgt_out_sequence, vocab)))
                f.write("IN  | {}\n".format(tensor_to_sentence(reference.insert_sequence, vocab)))
                f.write("DEL | {}\n".format(tensor_to_sentence(reference.delete_sequence, vocab)))
                for node in nodes:
                    f.write("CDT | {}\n".format(tensor_to_sentence(node.sequence, vocab)))
                f.write("===\n")
            f.write("\n")


def bleu_score(hypothesis: LongTensor, reference: LongTensor) -> float:
    """Compute the BLEU score of a hypothesis tensor with a reference tensor.

    Args:
        hypothesis (LongTensor): hypothesis / candidate sequence tensor of shape `(seq_len)`.
        reference (LongTensor): actual reference seuqence tensor of shape `(seq_len)`.

    Returns:
        float: the BLEU score between the two sequences.

    """
    stats = []
    for n in range(1, 5):
        h_ngrams = Counter(
            [tuple(hypothesis[i:i+n].tolist()) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i+n].tolist()) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((h_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))

    if 0 in stats:
        return 0

    c = len(hypothesis)
    r = len(reference)
    log_bleu_prec = sum([math.log(x / y) for x, y in zip(stats[::2], stats[1::2])]) / 4.
    return math.exp(min([0, 1 - r / c]) + log_bleu_prec)


def compute_bleu(model: EditTransformer, iterator: IteratorWrapper, limit: int, vocab: Vocab,
                 ex_writer: Optional[ExamplesWriter] = None, iteration: Optional[int] = None) -> float:
    """Compute the bleue score and return generated examples.

    Args:
        model (EditTransformer): the model being evaluated.
        iterator (IteratorWrapper): iterator yielding the batches to be evaluated.
        limit (int): the limit number of batch to go through in the iterator.
        vocab (Vocab): the vocabulary used with the corresponding model.
        ex_writer (Optional[ExamplesWriter]): an optional examples writer to write the examples generated.
        iteration (Optional[int]): an optional int corresponding to the iteration of the model.

    """
    nodes_list, references = beam_search(model, iterator, limit, vocab.stoi["<eos>"], vocab.stoi["<pad>"],
                                         draw_samples=False, draw_p=True)

    if ex_writer is not None:
        ex_writer.write_samples(iteration, nodes_list, references, vocab)

    bleus = []
    for nodes, reference in zip(nodes_list, references):
        bleus.append(bleu_score(nodes[0].sequence, reference.tgt_out_sequence))

    return sum(bleus) / len(bleus)


def evaluate_model(model: EditTransformer, train_iterator: IteratorWrapper, test_iterator: IteratorWrapper, limit: int,
                   vocab: Vocab, label: str, iteration: int, logger: Logger, tb_writter: SummaryWriter,
                   ex_writer: ExamplesWriter) -> None:
    """Evaluate a model over train and test iterator for a certain amount of batches.

    Args:
        model (EditTransformer): a trained model to evaluate.
        train_iterator (IteratorWrapper): an iterator with repeat False and shuffle True over the training samples.
        test_iterator (IteratorWrapper): an iterator with repeat False and shuffle True over the testing_samples.
        limit (int): limit the number of batches to evaluate in each iterators.
        vocab (Vocab): a vocab object used to ignore padding and to compute bleue_score.
        label (str): a label used for logging purpose.
        iteration (int): current training iteration being evaluated.
        logger (Logger): logger to use to log the results.
        tb_writter (SummaryWriter): tensorboard X summary writter with path already configured.
        ex_writer (ExamplesWriter): custom writer to write the examples.

    """
    model.eval()
    logger.info("Computing train losses...")
    train_loss = compute_loss(model, train_iterator, limit, vocab.stoi["<pad>"])
    logger.info("Done.")
    logger.info("Computing test losses...")
    test_loss = compute_loss(model, test_iterator, limit, vocab.stoi["<pad>"])
    logger.info("Done.")

    logger.info("Computing train bleu score...")
    train_bleu = compute_bleu(model, train_iterator, limit, vocab, ex_writer, iteration)
    logger.info("Done.")
    logger.info("Computing test bleu score...")
    test_bleu = compute_bleu(model, train_iterator, limit, vocab, ex_writer, iteration)
    logger.info("Done.")
    model.train()

    # logging
    logger.info("ITER #{}".format(iteration))
    logger.info("{}_train_loss: {}".format(label, train_loss))
    logger.info("{}_test_loss: {}".format(label, test_loss))
    logger.info("{}_train_bleu: {}".format(label, train_bleu))
    logger.info("{}_test_bleu: {}".format(label, test_bleu))

    # tb_logging
    tb_writter.add_scalar("{}_train_loss".format(label), train_loss, iteration)
    tb_writter.add_scalar("{}_test_loss".format(label), test_loss, iteration)
    tb_writter.add_scalar("{}_train_bleu".format(label), train_bleu, iteration)
    tb_writter.add_scalar("{}_test_bleu".format(label), test_bleu, iteration)
