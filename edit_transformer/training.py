from os.path import join
from csv import QUOTE_NONE
from logging import getLogger, Logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dependencies.text.torchtext.data.field import Field
from dependencies.text.torchtext.data.dataset import TabularDataset
from tensorboardX import SummaryWriter

from dependencies import data
from dependencies.config import Config
from dependencies.logger import setup_logging
from dependencies.workspace import IntegerDirectories
from edit_transformer.model import make_model
from edit_transformer.iterator import create_iterators
from edit_transformer.evaluation import evaluate_model, ExamplesWriter


def main(config: Config, logger: Logger, tb_writter: SummaryWriter, ex_writer: ExamplesWriter,
         device: torch.device, directory_path: str) -> None:
    """ Train an 'edit-transformer' model with the provided config.

    Args:
        config (Config): a config usually generated from file with the following keys / attributes:
            {
                'dataset.train_path (str)': absolute path to a preprocessed training dataset.
                'dataset.test_path (str)': absolute path to a preprocessed testing dataset.
                'dataset.valid_path (str)': absolute path to a preprocessed valid dataset.
                'vocab.max_size (int)': max size of the vocabulary (taking highest frequencies).
                'vocab.vector_name (str)': name of the vectors embeddings to use. (see an exhaustive list at:
                    https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab.Vocab.load_vectors)
                'training.batch_size (int)': batch size used on the training dataset.
                'training.num_iter (int)': number of training iteration to perform.
                'training.eval.batch_size (int)': batch size used during evaluation.
                'training.eval.small.threshold (int)': iteration threshold at which small eval are performed.
                'training.eval.small.limit (int)': maximum number of batch on which to run the small evaluation.
                'training.eval.big.threshold (int)': iteration threshold at which big eval are performed.
                'training.eval.big.limit (int)': maximum number of batch on which to run the big evaluation.
                'optimizer.lr (float)': learning rate to use.
                'model.edit_dim (int)': size of the edit vector to use.
                'model.n (int)': number of decoder and encoder layer to use.
                'model.d_ff (int)': size of the linear projection at the end of each decoder and encoder layer.
                'model.h (int)': number of multi-heads to use for multi-head attention (divisor of word_dim).
                'model.dropout (float)': dropout rate to use in the model.
                'lamb_reg (float)': dispersion regularization term of for the edit-encoder vMF distribution.
                'norm_eps (float)': epsilon used to sample the random noise added to the norm of the edit_vector.
                'norm_max (float)':scalar used to rescale the norm samples (corresponds to the maximum norm).
            }
        logger (Logger): the logger to use in the main function.
        tb_writter (SummaryWriter): tensorboardX writter with path already configured.
        ex_writer (ExamplesWriter): a writer to write the generated examples during beam search.
        device (torch.device): the torch device on which to run the training (cpu or gpu).
        directory_path (str): pass to this experiment directory.

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
    train_dataset = TabularDataset(config.dataset.train_path, "tsv", fields, csv_reader_params=csv_param)
    test_dataset = TabularDataset(config.dataset.test_path, "tsv", fields, csv_reader_params=csv_param)
    # valid_dataset = TabularDataset(config.dataset.valid_path, "tsv", fields, csv_reader_params=csv_param)
    logger.info("Done.")

    logger.info("Building Vocab...")
    field.build_vocab(train_dataset,
                      max_size=config.vocab.max_size,
                      specials=['<pad>', '<sos>', '<eos>'],
                      vectors=config.vocab.vector_name,
                      vectors_cache="/data/.vector_cache")
    logger.info("Done.")

    logger.info("Create Iterators...")
    datasets = (train_dataset, train_dataset, test_dataset)
    batch_sizes = (config.training.batch_size, config.training.eval.batch_size, config.training.eval.batch_size)
    repeats = (True, False, False)
    train_iterator, eval_train_iterator, eval_test_iterator = create_iterators(datasets, batch_sizes, repeats, device,
                                                                               field.vocab.stoi['<pad>'],
                                                                               field.vocab.stoi['<sos>'],
                                                                               field.vocab.stoi['<eos>'])
    logger.info("Done.")

    # 2. Model, optimizer, loss function initialization
    logger.info("Create Model...")
    embedding = nn.Embedding.from_pretrained(field.vocab.vectors, freeze=True)
    edit_transformer = make_model(embedding, config.model.edit_dim, config.model.n, config.model.d_ff, config.model.h,
                                  config.model.dropout, config.model.lamb_reg, config.model.norm_eps,
                                  config.model.norm_max)
    optimizer = Adam(edit_transformer.parameters(), lr=config.optimizer.lr)
    logger.info("Done.")

    # 3. Training loop.
    logger.info("Training...")
    edit_transformer.train()
    for batch in train_iterator:
        logits = edit_transformer(batch.src, batch.src_mask, batch.tgt_in, batch.tgt_mask, batch.insert,
                                  batch.insert_mask, batch.delete, batch.delete_mask)
        loss = F.nll_loss(F.log_softmax(logits.transpose(1, 2), dim=1), batch.tgt_out,
                          ignore_index=field.vocab.stoi['<pad>'])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del batch, logits, loss  # free up some space on the RAM (not used for evaluation).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free also the space on the GPU.

        iteration = train_iterator.iterator.iterations
        if iteration % config.training.eval.small.threshold == 0:
            evaluate_model(edit_transformer, eval_train_iterator, eval_test_iterator, config.training.eval.small.limit,
                           field.vocab, "small", iteration, logger, tb_writter, ex_writer)

        if iteration % config.training.eval.big.threshold == 0:
            evaluate_model(edit_transformer, eval_train_iterator, eval_test_iterator, config.training.eval.big.limit,
                           field.vocab, "big", iteration,  logger, tb_writter, ex_writer)

        if iteration % config.training.save_checkpoint_iter == 0:
            obj_to_save = {'step': iteration,
                           'emb_state_dict': embedding.state_dict(),
                           'model_state_dict': edit_transformer.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict()}
            torch.save(obj_to_save, join(directory_path, "checkpoint_{}.pth.tar".format(iteration)))

        if iteration == config.training.num_iter:
            break
    logger.info("Done.")


if __name__ == "__main__":
    # 0. Create folder for the training run
    integer_directories = IntegerDirectories(data.data_workspace.training_runs)
    directory_path_ = integer_directories.new_dir()

    # 1. logging
    # - Standard Logger
    setup_logging(root=directory_path_, noname=True)
    logger_ = getLogger(__name__)
    # - Tensorboard Logger
    tb_writter_ = SummaryWriter(directory_path_)
    # - Custom File Logger
    ex_writer_ = ExamplesWriter(join(directory_path_, "examples.txt"))

    # 2. config
    config_ = Config.from_file(join(data.code_workspace.configs, "edit_transformer", "edit_transformer.txt"))
    config_.to_file(join(directory_path_, "config.txt"))
    logger_.info("Config:\n{}".format(config_.to_str()))

    # 3. device
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. run the training
    main(config_, logger_, tb_writter_, ex_writer_, device_, directory_path_)
