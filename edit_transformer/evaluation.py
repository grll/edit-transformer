from typing import Tuple, List
from logging import Logger
from collections import Counter
import math

from torch import LongTensor
import torch.nn.functional as F
from dependencies.text.torchtext.vocab import Vocab
from tensorboardX import SummaryWriter

from edit_transformer.model import EditTransformer
from edit_transformer.iterator import IteratorWrapper
from edit_transformer.beam_decoder import beam_search


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


def compute_bleu(model: EditTransformer, iterator: IteratorWrapper, limit: int, vocab: Vocab) -> float:
    """Compute the bleue score and return generated examples.

    Args:
        model (EditTransformer): the model being evaluated.
    """
    nodes_list, references = beam_search(model, iterator, limit, vocab.stoi["<eos>"], vocab.stoi["<pad>"])

    bleus = []
    for nodes, reference in zip(nodes_list, references):
        bleus.append(bleu_score(nodes[0].sequence, reference.tgt_out_sequence))

    return sum(bleus) / len(bleus)


def evaluate_model(model: EditTransformer, train_iterator: IteratorWrapper, test_iterator: IteratorWrapper, limit: int,
                   vocab: Vocab, label: str, iteration: int, logger: Logger, tb_writter: SummaryWriter):
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

    """
    model.eval()
    logger.info("Computing train losses...")
    train_loss = compute_loss(model, train_iterator, limit, vocab.stoi["<pad>"])
    logger.info("Done.")
    logger.info("Computing test losses...")
    test_loss = compute_loss(model, test_iterator, limit, vocab.stoi["<pad>"])
    logger.info("Done.")

    logger.info("Computing train bleu score...")
    train_bleu = compute_bleu(model, train_iterator, limit, vocab)
    logger.info("Done.")
    logger.info("Computing test bleu score...")
    test_bleu = compute_bleu(model, train_iterator, limit, vocab)
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
