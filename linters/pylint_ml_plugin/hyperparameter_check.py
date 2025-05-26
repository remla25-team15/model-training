from pylint.checkers import BaseChecker


class HyperparameterExplicitSetChecker(BaseChecker):

    name = "hyperparameter-explicit-set-checker"
    priority = -1
    msgs = {
        "W9003": (
            "Model %s instantiated without explicitly setting hyperparameter(s): %s",
            "hyperparameter-not-explicitly-set",
            "Key hyperparameters should be explicitly set to avoid default assumptions.",
        ),
    }

    model_hyperparams = {
        "GaussianNB": {"var_smoothing"},
    }

    def visit_call(self, node):
        # Detect the class name being instantiated
        try:
            func_name = node.func.as_string()
        except AttributeError:
            return

        class_name = func_name.split(".")[-1]

        if class_name not in self.model_hyperparams:
            return

        explicit_params = {kw.arg for kw in node.keywords}

        missing = self.model_hyperparams[class_name] - explicit_params

        if missing:
            self.add_message(
                "hyperparameter-not-explicitly-set",
                node=node,
                args=(class_name, ", ".join(sorted(missing))),
            )
