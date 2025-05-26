from pylint.checkers import BaseChecker


class HardcodedDataPathChecker(BaseChecker):

    name = "hardcoded-data-path-checker"
    priority = -1
    msgs = {
        "W9001": (
            "Hardcoded data path detected: %s",
            "hardcoded-data-path",
            "Avoid hardcoding file paths, especially to training data.",
        ),
    }

    def visit_call(self, node):
        try:
            func_name = node.func.as_string()
        except AttributeError:
            return

        if func_name in ("open", "pandas.read_csv", "pd.read_csv"):
            for arg in node.args:
                if hasattr(arg, "value") and isinstance(arg.value, str):
                    if "/" in arg.value or "\\" in arg.value:
                        # Pass the arg node instead of the call node
                        self.add_message(
                            "hardcoded-data-path", node=arg, args=(arg.value,)
                        )
                elif arg.as_string().startswith(("'", '"')) and (
                    "/" in arg.as_string() or "\\" in arg.as_string()
                ):
                    self.add_message(
                        "hardcoded-data-path", node=arg, args=(arg.as_string(),)
                    )
