from os.path import join
from csv import QUOTE_NONE
from logging import getLogger, Logger
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchtext.data.field import Field
from torchtext.data.dataset import TabularDataset
from torchtext.data.iterator import BucketIterator

from dependencies import data
from dependencies.config import Config
from dependencies.logger import setup_logging
from edit_transformer.model import make_model
from edit_transformer.iterator import IteratorWrapper


def main(config: Config, logger: Logger, device: torch.device) -> None:
    """ Train an 'edit-transformer' model with the provided config.

    Args:
        config (Config): a config usually generated from file with the following keys / attributes: {
            'dataset.train_path (str)': absolute path to a preprocessed training dataset.
            'dataset.test_path (str)': absolute path to a preprocessed testing dataset.
            'dataset.valid_path (str)': absolute path to a preprocessed valid dataset.
            'vocab.max_size (int)': max size of the vocabulary (taking highest frequencies).
            'vocab.vector_name (str)': name of the vectors embeddings to use. (see an exhaustive list at:
                https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab.Vocab.load_vectors)
            'training.batch_size (int)': batch size used on the training dataset.
        }
        logger (Logger): the logger to use in the main function.
        device (torch.device): the torch device on which to run the training (cpu or gpu)

    """
    # 1. Data Processing
    # Create the Fields
    logger.info("Creating the fields...")
    field = Field(lower=True, include_lengths=True, batch_first=True)
    fields = [("src", field), ("tgt", field), ("insert", field), ("delete", field)]
    logger.info("Done.")

    # Load the datasets
    logger.info("Loading Datasets...")
    csv_param = {'quoting': QUOTE_NONE}
    # train_dataset = TabularDataset(config.dataset.train_path, "tsv", fields, csv_reader_params=csv_param)
    # test_dataset = TabularDataset(config.dataset.test_path, "tsv", fields, csv_reader_params=csv_param)
    valid_dataset = TabularDataset(config.dataset.valid_path, "tsv", fields, csv_reader_params=csv_param)
    logger.info("Done.")

    logger.info("Building Vocab...")
    field.build_vocab(valid_dataset,
                      max_size=config.vocab.max_size,
                      specials=['<pad>', '<sos>', '<eos>'],
                      vectors=config.vocab.vector_name,
                      vectors_cache="/data/.vector_cache")
    logger.info("Done.")

    logger.info("Create Iterator...")
    iterator = IteratorWrapper(BucketIterator(valid_dataset, batch_size=config.training.batch_size, device=device),
                               field.vocab.stoi['<pad>'], field.vocab.stoi['<sos>'], field.vocab.stoi['<eos>'])
    logger.info("Done.")

    # 2. Model, optimizer, loss function initialization
    embedding = nn.Embedding.from_pretrained(field.vocab.vectors, freeze=True)
    edit_transformer = make_model(embedding)

    # 3. Training loop.
    for batch in iterator:
        out_prob = edit_transformer(batch.src, batch.src_mask, batch.tgt_in, batch.tgt_mask, batch.insert, batch.delete)
        # loss = nll(out_prob, batch.tgt_out)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()


if __name__ == "__main__":
    # 1. logging
    setup_logging()
    logger_ = getLogger(__name__)
    # 2. config
    config_ = Config.from_file(join(data.code_workspace.configs, "edit_transformer", "edit_transformer.txt"))
    # 3. device
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 4. run the training
    main(config_, logger_, device_)
