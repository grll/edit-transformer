from typing import Dict, List, Any
from logging import Logger, getLogger
from os.path import join


import torch
import torch.nn as nn
import spacy

from dependencies import data
from dependencies.typing import T_LongTensor
from dependencies.config import Config
from dependencies.logger import setup_logging
from dependencies.text.torchtext.vocab import Vocab
from edit_transformer.model import make_model
from edit_transformer.iterator import Batch
from edit_transformer.beam_decoder import beam_search
from edit_transformer.evaluation import tensor_to_sentence


def preprocess(prompt: str, nlp: Any, vocab: Vocab) -> T_LongTensor:
    """Preprocess the prompt from the user and return an input tensor for the model

    Args:
        prompt (str): The sentence input prompted by the user.
        nlp (Any): A spacy model to use to preprocess the data (tokenization and entities replacement).
        vocab (Vocab): A vocabulary object to use to convert the tokenized sentence into a tensor.

    Returns:
        T_LongTensor: A long tensor of shape `(1, seq_len)` corresponding to the tokenized sentence.

    """
    preprocessed_tokens = []
    entities: Dict[str, List[str]] = {}

    doc = nlp(prompt)
    for token in doc:
        ent_name = "<" + token.ent_type_.lower() + ">"
        if token.ent_iob_ in ["O", ""]:
            preprocessed_tokens.append(token.text.lower())
        elif token.ent_iob_ == "B":
            if ent_name not in entities:
                entities[ent_name] = []
            entities[ent_name].append(token.text)
            preprocessed_tokens.append(ent_name)
        elif token.ent_iob_ == "I":
            entities[ent_name][-1] += " " + token.text

    preprocessed_tokens.append("<eos>")

    return torch.tensor([[vocab.stoi[t] for t in preprocessed_tokens]])  # TODO: handle the device here as well




def main(config: Config, logger: Logger) -> None:
    """Perform the Generation of an input using the provided Config file.

    Args:
        config (Config): the config used to run the generation must contain the folowwing keys/attributes:
            {
                'model.vocab_path (str)': path to the vocab to use to create the model.
                'model.checkpoint_path (str)': path to the model checkpoint to load.
                'model.config_path (str)': path to the training config used (defining the model in use).
                'beam_search.q_limit (int)': maximum number of node in a queue of the beam search (limit computation).
                'beam_search.attempts (int)': number of random beam serach to perform.
            }
        logger (Logger): the logger to use in this script.

    """
    logger.info("Loading pre-trained model...")
    m_config = Config.from_file(config.model.config_path)
    vocab = torch.load(config.model.vocab_path)

    embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=True)
    edit_transformer = make_model(embedding, m_config.model.edit_dim, m_config.model.n, m_config.model.d_ff,
                                  m_config.model.h, m_config.model.dropout, m_config.model.lamb_reg,
                                  m_config.model.norm_eps, m_config.model.norm_max)

    obj = torch.load(config.model.checkpoint_path, map_location="cpu")  # TODO: use GPU if available.
    edit_transformer.load_state_dict(obj['model_state_dict'])
    logger.info("Done.")

    logger.info("Loading Pre-processing model...")
    nlp = spacy.load("en_core_web_lg")
    logger.info("Done.")

    while True:
        prompt = input("Enter a sentence to be augmented:")
        logger.info("Augmenting sentence: `{}`...".format(prompt))

        logger.info("Pre-processing the sentence...")
        src = preprocess(prompt, nlp, vocab)  # TODO: specify the device.
        logger.info("Done.")

        logger.info("Creating a Batch for the model...")
        batch = Batch.from_src_seq_only(src, vocab.stoi["<pad>"], vocab.stoi["<sos>"], vocab.stoi["<eos>"])
        logger.info("Done.")


        logger.info("Performing the beam search...")
        nodes_list_f = []
        reference_f = []
        for i in range(config.beam_search.attempts):
            nodes_list, references = beam_search(edit_transformer, batch, vocab.stoi["<eos>"], vocab.stoi["<pad>"],
                                                 q_limit=config.beam_search.q_limit, draw_samples=True, draw_p=True)
            nodes_list_f.extend(nodes_list[0])
            reference_f = references[0]
            logger.info("{}/{}".format(i+1, config.beam_search.attempts))

        nodes_list_f.sort(key=lambda x: x.queue_score())
        logger.info("Done.")

        logger.info("IN | {}".format(tensor_to_sentence(reference_f.src_sequence, vocab)))
        for node in nodes_list_f[:5]:
            logger.info("CDT | {:.4f} | {:.4f} | {}".format(node.proba, node.queue_score(),
                                                            tensor_to_sentence(node.sequence, vocab)))


if __name__ == "__main__":
    # 1. Logging
    setup_logging()
    logger_ = getLogger(__name__)

    # 2. Config
    config_ = Config.from_file(join(data.code_workspace.configs, "edit_transformer", "generation.txt"))
    logger_.info("Config:\n{}".format(config_.to_str()))

    #3. Run the script
    main(config_, logger_)
