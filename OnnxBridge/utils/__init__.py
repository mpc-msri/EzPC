from enum import Enum

from utils.logger import setup_logger

# Logger Setup
logger = setup_logger()


class Party(Enum):
    """
    Class to define two Party.
    """

    BOB = 0  # server
    ALICE = 1  # client


class VariableGen:
    """
    Class to generate new variable names.
    """

    number = 0
    counter = 0
    reshape_counter = 0

    @classmethod
    def get_var(cls):
        VariableGen.number += 1
        return f"var{VariableGen.number}"

    @classmethod
    def get_loop_var(cls):
        VariableGen.counter += 1
        return f"i{VariableGen.counter}"

    @classmethod
    def get_reshape_var(cls):
        VariableGen.reshape_counter += 1
        return f"re{VariableGen.reshape_counter}"

    @classmethod
    def reset_loop_var_counter(cls):
        VariableGen.counter = 0


def support_device(device: str):
    if device == "2PC":
        return True
    logger.exception("Currently only 2PC devices are supported")
