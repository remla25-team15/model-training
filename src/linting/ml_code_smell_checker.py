from pylint.checkers import BaseChecker

class MLCodeSmellChecker(BaseChecker):
    name = "ml-code-smell"
    msgs = {
        "C9001": (
            "Hardcoded hyperparameter detected: %s",
            "hardcoded-hyperparameter",
            "Avoid hardcoding hyperparameters; use configuration files or constants.",
        ),
        "C9002": (
            "Missing random seed for reproducibility",
            "missing-random-seed",
            "Set a random seed to ensure reproducibility in ML experiments.",
        ),
    }

    def visit_call(self, node):
        # Detect hardcoded learning rate
        if (
            node.func.as_string() in ["sklearn.linear_model.SGDClassifier", "torch.optim.SGD"]
            and "lr" in [kw.arg for kw in node.keywords]
        ):
            self.add_message("hardcoded-hyperparameter", node=node, args=("learning rate",))

        # Detect missing random seed
        if node.func.as_string() in ["random.seed", "numpy.random.seed", "torch.manual_seed"]:
            return
        if node.func.as_string() in ["sklearn.model_selection.train_test_split"]:
            self.add_message("missing-random-seed", node=node)

def register(linter):
    linter.register_checker(MLCodeSmellChecker(linter))