from pylint.checkers import BaseChecker


class MissingRandomSeedChecker(BaseChecker):

    name = "missing-random-seed-checker"
    priority = -1
    msgs = {
        "W9002": (
            "No random seed set for module(s): %s",
            "missing-random-seed",
            "Random seed not set; results may be nondeterministic.",
        ),
    }

    # Map aliases to canonical module names
    ALIAS_MAP = {
        "np": "numpy",
        "tf": "tensorflow",
    }

    def __init__(self, linter=None):
        super().__init__(linter)
        self.imported_modules = set()
        self.seed_set_modules = set()
        self.module_node = None

    def visit_module(self, node):
        self.module_node = node

    def visit_import(self, node):
        for name, alias in node.names:
            self.imported_modules.add(name)
            if alias:
                self.imported_modules.add(alias)

    def visit_importfrom(self, node):
        if node.modname:
            self.imported_modules.add(node.modname)
            for name, alias in node.names:
                if alias:
                    self.imported_modules.add(alias)
                else:
                    self.imported_modules.add(name)

    def visit_call(self, node):
        try:
            func_name = node.func.as_string()
        except AttributeError:
            return

        if func_name in (
            "random.seed",
            "numpy.random.seed",
            "np.random.seed",
            "torch.manual_seed",
            "tensorflow.random.set_seed",
            "tf.random.set_seed",
        ):
            base = func_name.split(".")[0]
            self.seed_set_modules.add(base)

    def _normalize_modules(self, modules):
        """Convert aliases to canonical module names."""
        normalized = set()
        for mod in modules:
            normalized.add(self.ALIAS_MAP.get(mod, mod))
        return normalized

    def close(self):
        modules_requiring_seed = {"random", "numpy", "torch", "tensorflow"}

        normalized_imports = self._normalize_modules(self.imported_modules)
        normalized_seeds = self._normalize_modules(self.seed_set_modules)

        used_without_seed = (
            normalized_imports & modules_requiring_seed
        ) - normalized_seeds

        if used_without_seed:
            self.add_message(
                "missing-random-seed",
                node=self.module_node,
                args=(", ".join(sorted(used_without_seed)),),
            )
