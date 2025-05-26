from .hardcoded_data_path import HardcodedDataPathChecker
from .hyperparameter_check import HyperparameterExplicitSetChecker
from .missing_random_seed import MissingRandomSeedChecker


def register(linter):
    linter.register_checker(HardcodedDataPathChecker(linter))
    linter.register_checker(MissingRandomSeedChecker(linter))
    linter.register_checker(HyperparameterExplicitSetChecker(linter))
