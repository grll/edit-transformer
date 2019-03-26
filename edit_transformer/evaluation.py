from typing import Tuple, List
from logging import Logger

import torch.nn.functional as F
from dependencies.text.torchtext.vocab import Vocab
from tensorboardX import SummaryWriter

from edit_transformer.model import EditTransformer
from edit_transformer.iterator import IteratorWrapper


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


def compute_bleu(model: EditTransformer, iterator: IteratorWrapper, limit: int,
                 vocab: Vocab) -> Tuple[float, List[str]]:
    """Compute the bleue score and return generated examples."""
    return 0.0, []


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
    train_loss = compute_loss(model, train_iterator, limit, vocab.stoi["<pad>"])
    test_loss = compute_loss(model, test_iterator, limit, vocab.stoi["<pad>"])

    train_bleu, train_examples = compute_bleu(model, train_iterator, limit, vocab)
    test_bleu, test_examples = compute_bleu(model, train_iterator, limit, vocab)
    model.train()

    # logging
    logger.info("ITER #{}".format(iteration))
    logger.info("{}_train_loss: {}".format(label, train_loss))
    logger.info("{}_test_loss: {}".format(label, test_loss))

    # tb_logging
    tb_writter.add_scalar("{}_train_loss".format(label), train_loss, iteration)
    tb_writter.add_scalar("{}_test_loss".format(label), test_loss, iteration)
