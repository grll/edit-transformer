from logging import Logger, getLogger
from os.path import join


from dependencies import data
from dependencies.config import Config
from dependencies.logger import setup_logging
from edit_transformer.model import EditTransformer


def main(config: Config, logger: Logger) -> None:
    """Perform the Generation of an input using the provided Config file.

    Args:
        config (Config): the config used to run the generation must contain the folowwing keys/attributes:
            {
                'config.model.checkpoint_path (str)': path to the model checkpoint to load.
                ''

            }
        logger (Logger): the logger to use in this script.

    """
    # Load the model
    # get prompt from user.
    # preprocess the prompt from user.
    # run it through the model.
    # output finished sentences from the beam search.

if __name__ == "__main__":
    # 1. logging
    setup_logging()
    logger_ = getLogger(__name__)

    # 2. config
    config_ = Config.from_file(join(data.code_workspace.configs, "edit_transformer", "generation.txt"))
    logger_.info("Config:\n{}".format(config_.to_str()))

