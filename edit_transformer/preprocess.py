from typing import Optional
from os.path import join

from dependencies import data
from dependencies.config import Config


def process_line(line: str, free_set: Optional[set] = None) -> str:
    """ Process a line of the original file (TSV format of pair of sentences)

    Args:
        line (str): a line of 2 sentences separated with a "\t" token and with a "\n" token at the end.
        free_set (Optional[set]): an optional set of string to remove from the insert and delete set.

    Returns:
        str: the output line to print in the generated file with "\n" token at the end.

    """
    if free_set is None:
        free_set = set()

    # clean the line to avoid "" tokens.
    for t in [" \n", "\n"]:
        line = line.replace(t, "")
    for t in [" \t ", " \t", "\t "]:
        line = line.replace(t, "\t")

    assert len(line.split("\t")) == 2
    sentence_1, sentence_2 = line.lower().split("\t")
    set_1 = set(sentence_1.split(" "))
    set_2 = set(sentence_2.split(" "))

    insert = (set_2 - set_1 - free_set)
    delete = (set_1 - set_2 - free_set)

    return sentence_1 + "\t" + sentence_2 + "\t" + " ".join(insert) + "\t" + " ".join(delete) + "\n"


def main(config: Config) -> None:
    """Preprocess TSV formatted files specified in config and create new TSV preprocessed corresponding files.

    Notes:
        The files specified must be in the following format `sentence_1 \t sentence_2` and will be filtered and
        preprocessed into the following format `sentence_1 \t sentence_2 \t insert \t delete`.

    Args:
        config (Config): A config object (usually loaded from file) that possess the following keys/attributes:
            - raw_data_paths (List[str]): path to original TSV files.
            - use_free_set (bool): weather to filter `insert` and `delete` using the specified `free_set.txt` file.
`           - free_set_path (str): path to the `free_set.txt` file (one token per line).

    """
    if config.use_free_set:
        with open(config.free_set_path, mode="r", encoding="utf8") as f:
            free_set = set([line.strip().lower() for line in f])
    else:
        free_set = None

    for in_path in config.raw_data_paths:
        split_path = in_path.split(".")
        out_path = ".".join(split_path[:-1]) + "_preprocessed_free_set_{}.{}".format(config.use_free_set,
                                                                                     split_path[-1])
        with open(in_path, mode="r", encoding="utf8") as f_in:
            with open(out_path, mode="w", encoding="utf8") as f_out:
                for line in f_in:
                    out = process_line(line, free_set)
                    f_out.write(out)


if __name__ == "__main__":
    config_ = Config.from_file(join(data.code_workspace.configs, "edit_transformer", "preprocess.txt"))
    main(config_)
